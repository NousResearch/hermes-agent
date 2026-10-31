import Foundation
import Observation
import HermesCore

@MainActor public protocol CompanionTransport: AnyObject {
    var onEvent: ((JSON) -> Void)? { get set }
    var onDisconnect: ((Error) -> Void)? { get set }
    func connect(endpoint: Endpoint, username: String, password: String) async throws
    func reconnect() async throws
    func request(_ method: String, params: JSON) async throws -> JSON
    func http(_ path: String, method: String, body: JSON?, query: [String: String]) async throws -> JSON
    func disconnect()
    func signOut()
}

extension Gateway: CompanionTransport {}

@MainActor @Observable public final class CompanionStore {
    public var session: SessionState?
    public var errorMessage: String?
    public private(set) var isConnected = false { didSet { syncServices() } }
    public private(set) var connecting = false
    public private(set) var loadingSession = false
    public private(set) var controlBusy = false
    public var isPreview = false { didSet { syncServices() } }
    public var controlsCompatible: Bool { session?.contractSupported != false }
    public var host = ""
    public var selectedProfile = "" { didSet { syncServices() } }
    public var profiles: [JSON] = []
    public var sessions: [JSON] = []
    public var modelOverride = ""
    public var currentModel = ""
    public var boards: [JSON] = []
    public var selectedBoard = ""
    public var columns: [JSON] = []
    public private(set) var boardBusy = false
    public private(set) var boardNeedsRefresh = true
    public var boardUpdatedAt: Date?
    private let transport: CompanionTransport
    public let library: WorkspaceLibrary
    public let scheduledJobs: ScheduledJobsStore
    private var connectionVersion = UUID()
    private var selectionVersion = UUID()
    private var boardVersion = UUID()
    private var resumeEvents: [JSON] = []

    public init(transport: CompanionTransport) {
        self.transport = transport
        self.library = WorkspaceLibrary(transport: transport)
        self.scheduledJobs = ScheduledJobsStore(transport: transport)
        transport.onEvent = { [weak self] event in self?.receive(event) }
        transport.onDisconnect = { [weak self] error in self?.connectionLost(error) }
    }
    private func syncServices() {
        library.configure(connected: isConnected && !isPreview, profile: selectedProfile)
        scheduledJobs.configure(connected: isConnected && !isPreview, profile: selectedProfile)
    }
    private func scope(_ values: [String: JSON] = [:]) -> JSON {
        var result = values
        if !selectedProfile.isEmpty { result["profile"] = .string(selectedProfile) }
        return .object(result)
    }
    public func connect(url: String, username: String, password: String) async {
        guard !connecting, !isPreview else { return }
        signOut()
        let version = connectionVersion
        connecting = true; errorMessage = nil
        do {
            let endpoint = try Endpoint(url)
            try await transport.connect(endpoint: endpoint, username: username, password: password)
            guard version == connectionVersion else { return }
            host = endpoint.baseURL.host ?? "Mac"; isConnected = true; connecting = false
            await refreshRoster()
        } catch {
            guard version == connectionVersion else { return }
            connecting = false; isConnected = false; errorMessage = error.localizedDescription
        }
    }
    public func reconnect() async {
        guard !connecting, !isPreview else { return }
        let savedID = session?.storedID
        suspend()
        let version = connectionVersion
        let selection = selectionVersion; let profile = selectedProfile
        connecting = true; errorMessage = nil
        do {
            try await transport.reconnect()
            guard version == connectionVersion else { return }
            isConnected = true; connecting = false
            await refreshRoster()
            if version == connectionVersion, selection == selectionVersion, profile == selectedProfile,
               let id = savedID { await openSession(id) }
        } catch {
            guard version == connectionVersion else { return }
            connecting = false; errorMessage = error.localizedDescription
        }
    }
    public func suspend() {
        connectionVersion = UUID(); selectionVersion = UUID(); boardVersion = UUID()
        connecting = false; isConnected = false; loadingSession = false
        controlBusy = false; boardBusy = false; boardNeedsRefresh = true
        session?.disconnect(); resumeEvents.removeAll(); transport.disconnect()
    }
    public func signOut() {
        suspend(); transport.signOut()
        session = nil; profiles = []; sessions = []; boards = []; columns = []
        selectedProfile = ""; selectedBoard = ""; host = ""; currentModel = ""
        boardUpdatedAt = nil; modelOverride = ""; errorMessage = nil
        isPreview = false
    }
    #if DEBUG
    public func beginPreview() {
        // Invalidate a pending login before any sample state is installed.
        signOut()
        isPreview = true
        selectedProfile = "Preview"
    }
    #endif
    private func connectionLost(_ error: Error) {
        suspend(); errorMessage = "Connection lost. The Mac may still be working. Reconnect and refresh before retrying."
    }
    public func refreshRoster() async {
        guard isConnected, !isPreview else { return }
        let version = connectionVersion
        let selection = selectionVersion
        do {
            let result = try await transport.request("profiles.list", params: .object(["include_sessions": .bool(false)]))
            guard version == connectionVersion, selection == selectionVersion else { return }
            guard let rows = result["profiles"].array else { throw FeatureError.invalid("Invalid agent list from Hermes.") }
            profiles = rows
            await refreshSessions()
        } catch { if version == connectionVersion { errorMessage = error.localizedDescription } }
    }
    public func refreshSessions() async {
        guard isConnected, !isPreview else { return }
        let version = selectionVersion; let connection = connectionVersion
        do {
            let result = try await transport.request("session.list", params: scope(["limit": .number(200)]))
            guard version == selectionVersion, connection == connectionVersion else { return }
            guard let rows = result["sessions"].array else { throw FeatureError.invalid("Invalid conversation list from Hermes.") }
            sessions = rows
        } catch { if version == selectionVersion, connection == connectionVersion { errorMessage = error.localizedDescription } }
    }
    public func chooseProfile(_ name: String) async {
        guard !isPreview, !controlBusy else { return }
        selectionVersion = UUID(); selectedProfile = name; session = nil; sessions = []; currentModel = ""; modelOverride = ""
        loadingSession = false; resumeEvents.removeAll(); errorMessage = nil
        await refreshSessions()
    }
    public func openSession(_ id: String) async {
        guard isConnected, !isPreview, !controlBusy else { return }
        selectionVersion = UUID()
        let version = selectionVersion; let connection = connectionVersion
        loadingSession = true; session = nil; resumeEvents.removeAll(); errorMessage = nil
        do {
            let result = try await transport.request("session.resume", params: scope(["session_id": .string(id), "lazy": .bool(true)]))
            guard version == selectionVersion, connection == connectionVersion else { return }
            session = try SessionState(snapshot: result)
            currentModel = result["info"]["model"].string ?? "Profile default"
            loadingSession = false
            // Snapshot and stream are not atomic. Do not concatenate buffered deltas
            // onto snapshot history; wait for full completion or an explicit refresh.
            let relevant = resumeEvents.filter { $0["session_id"].string == session?.sessionID }
            resumeEvents.removeAll()
            if !relevant.isEmpty {
                session?.needsRefresh = true
                session?.activity = "Conversation changed while loading · refresh"
            }
            await refreshApprovals(version: version, connection: connection)
        } catch {
            guard version == selectionVersion, connection == connectionVersion else { return }
            loadingSession = false; errorMessage = error.localizedDescription
        }
    }
    public func createChat(title: String = "", cwd: String = "", provider: String = "") async {
        guard isConnected, controlsCompatible, !isPreview, !loadingSession, !controlBusy else { return }
        selectionVersion = UUID()
        let version = selectionVersion; let connection = connectionVersion
        loadingSession = true; errorMessage = nil; session = nil
        var params: [String: JSON] = ["source": .string("web"), "close_on_disconnect": .bool(false)]
        if !title.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { params["title"] = .string(title) }
        if !cwd.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { params["cwd"] = .string(cwd) }
        if !modelOverride.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { params["model"] = .string(modelOverride) }
        if !provider.isEmpty { params["provider"] = .string(provider) }
        do {
            let result = try await transport.request("session.create", params: scope(params))
            guard version == selectionVersion, connection == connectionVersion else { return }
            session = try SessionState(snapshot: result); loadingSession = false
            currentModel = result["info"]["model"].string ?? "Profile default"
        } catch {
            guard version == selectionVersion, connection == connectionVersion else { return }
            loadingSession = false; errorMessage = "Could not confirm new conversation. Refresh the list before retrying. " + error.localizedDescription
        }
    }
    public func describeProfile(_ name: String) async -> JSON? {
        guard isConnected, !isPreview, profiles.contains(where: { $0["name"].string == name }) else { return nil }
        let connection = connectionVersion
        do {
            let result = try await transport.request("profiles.describe", params: .object(["name": .string(name)]))
            guard connection == connectionVersion else { return nil }
            guard result["name"].string == name else { throw FeatureError.invalid("Profile response identity changed.") }
            return result
        } catch { if connection == connectionVersion { errorMessage = error.localizedDescription }; return nil }
    }
    public func send(_ text: String) async {
        let text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard isConnected, !isPreview, !controlBusy, !text.isEmpty, session?.canSend == true,
              let sid = session?.sessionID else { return }
        let version = selectionVersion; let connection = connectionVersion
        session?.beginSend(text); errorMessage = nil
        do {
            let result = try await transport.request("prompt.submit", params: scope(["session_id": .string(sid), "text": .string(text)]))
            guard version == selectionVersion, connection == connectionVersion else { return }
            guard ["streaming", "queued"].contains(result["status"].string ?? "") else {
                throw FeatureError.invalid("Unexpected send acknowledgement.")
            }
            session?.markAccepted()
        } catch {
            guard version == selectionVersion, connection == connectionVersion else { return }
            session?.markUncertain()
            errorMessage = "Send result unknown. Refresh this conversation before retrying. " + error.localizedDescription
        }
    }
    public func stop() async {
        guard let sid = session?.sessionID else { return }
        await sessionControl(RPCCommand(method: "session.interrupt", params: scope(["session_id": .string(sid)])))
    }
    public func respond(_ request: PendingRequest, answer: String) async {
        guard session?.requests.contains(request) == true, let sid = session?.sessionID else {
            errorMessage = "This request is no longer current. Refresh before responding."; return
        }
        do {
            let command = try Commands.respond(request, sessionID: sid, answer: answer)
            await sessionControl(command, requestID: request.id)
        } catch { errorMessage = error.localizedDescription }
    }
    private func sessionControl(_ command: RPCCommand, requestID: String? = nil) async {
        guard isConnected, !isPreview, !loadingSession, !controlBusy, session?.needsRefresh == false else { return }
        let version = selectionVersion; let connection = connectionVersion
        controlBusy = true; errorMessage = nil
        do {
            let result = try await transport.request(command.method, params: command.params)
            guard version == selectionVersion, connection == connectionVersion else { return }
            if command.method == "approval.respond", result["resolved"].int != 1 {
                throw FeatureError.invalid("Approval was not resolved. It may have expired or changed.")
            }
            if command.method == "clarify.respond", result["status"].string != "ok" {
                throw FeatureError.invalid("Question is no longer pending. Refresh the conversation.")
            }
            if command.method == "session.interrupt", result["status"].string != "interrupted" {
                throw FeatureError.invalid("Stop was not acknowledged by Hermes.")
            }
            if let id = requestID { session?.requests.removeAll { $0.id == id } }
            if command.method == "session.interrupt", session?.running == true {
                session?.activity = "Stop requested · waiting for Mac"
            }
            controlBusy = false
        } catch {
            guard version == selectionVersion, connection == connectionVersion else { return }
            controlBusy = false; session?.needsRefresh = true; session?.requests.removeAll()
            errorMessage = "Control result not confirmed. Refresh before retrying. " + error.localizedDescription
        }
    }
    private func refreshApprovals(version: UUID, connection: UUID) async {
        guard let sid = session?.sessionID else { return }
        do {
            let result = try await transport.request("approval.pending", params: scope(["session_id": .string(sid)]))
            guard version == selectionVersion, connection == connectionVersion else { return }
            for item in result["approvals"].array ?? [] { session?.addPending(item, kind: .approval) }
        } catch {
            if version == selectionVersion, connection == connectionVersion { errorMessage = "Approval list unavailable. Use the Mac for approvals until refreshed." }
        }
    }
    private func receive(_ event: JSON) {
        guard isConnected else { return }
        if loadingSession { if resumeEvents.count < 512 { resumeEvents.append(event) }; return }
        if event["type"].string == "gateway.ready", session != nil {
            session?.needsRefresh = true; session?.requests.removeAll(); return
        }
        session?.apply(event)
    }
    public func refreshBoards() async {
        guard isConnected, !isPreview, !boardBusy else { return }
        let connection = connectionVersion
        do {
            let result = try await transport.http("/api/plugins/kanban/boards", method: "GET", body: nil, query: [:])
            guard connection == connectionVersion else { return }
            guard let rows = result["boards"].array else { throw FeatureError.invalid("Kanban is unavailable on this gateway.") }
            boards = rows
            if selectedBoard.isEmpty { selectedBoard = result["current"].string ?? rows.first?["slug"].string ?? "" }
            await refreshBoard()
        } catch { if connection == connectionVersion { errorMessage = error.localizedDescription; boardNeedsRefresh = true } }
    }
    public func selectBoard(_ slug: String) async {
        guard !boardBusy, !isPreview else { return }
        boardVersion = UUID(); selectedBoard = slug; columns = []; boardUpdatedAt = nil; boardNeedsRefresh = true
        await refreshBoard()
    }
    public func refreshBoard() async {
        guard isConnected, !isPreview, !selectedBoard.isEmpty, !boardBusy else { return }
        let board = selectedBoard; let version = boardVersion; let connection = connectionVersion
        boardBusy = true
        do {
            let result = try await transport.http("/api/plugins/kanban/board", method: "GET", body: nil, query: ["board": board])
            guard version == boardVersion, connection == connectionVersion else { return }
            guard let rows = result["columns"].array else { throw FeatureError.invalid("Invalid Kanban response. Existing cards may be stale.") }
            columns = rows; boardUpdatedAt = Date(); boardNeedsRefresh = false; boardBusy = false
        } catch {
            guard version == boardVersion, connection == connectionVersion else { return }
            boardBusy = false; boardNeedsRefresh = true; errorMessage = error.localizedDescription
        }
    }
    public func createTask(title: String, body: String, assignee: String) async {
        do { try await mutateTask(path: "/api/plugins/kanban/tasks", method: "POST", body: Commands.createTask(title: title, body: body, assignee: assignee)) }
        catch { errorMessage = error.localizedDescription }
    }
    public func updateTask(original: JSON, title: String, body: String, assignee: String, status: String) async {
        do {
            guard let id = original["id"].string else { throw FeatureError.invalid("Missing task identity. Refresh the board.") }
            guard !title.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { throw FeatureError.invalid("A task needs a title.") }
            let fields = ["title": title, "body": body, "assignee": assignee, "status": status]
            let changed = fields.filter { (original[$0.key].string ?? "") != $0.value }.mapValues { JSON.string($0) }
            guard !changed.isEmpty else { errorMessage = nil; return }
            try await mutateTask(path: Commands.taskPath(id), method: "PATCH", body: .object(changed), expectedID: id)
        } catch { errorMessage = error.localizedDescription }
    }
    private func mutateTask(path: String, method: String, body: JSON, expectedID: String? = nil) async throws {
        guard isConnected, controlsCompatible, !isPreview, !selectedBoard.isEmpty, !boardBusy, !boardNeedsRefresh else {
            throw FeatureError.invalid("Connect and refresh the selected board before changing tasks.")
        }
        let connection = connectionVersion; let version = boardVersion; let board = selectedBoard
        boardBusy = true; errorMessage = nil
        do {
            let result = try await transport.http(path, method: method, body: body, query: ["board": board])
            guard connection == connectionVersion, version == boardVersion else { return }
            guard let id = result["task"]["id"].string, !id.isEmpty,
                  expectedID == nil || expectedID == id else { throw FeatureError.invalid("Missing or mismatched task receipt.") }
            boardBusy = false; await refreshBoard()
        } catch {
            guard connection == connectionVersion, version == boardVersion else { return }
            boardBusy = false; boardNeedsRefresh = true
            throw FeatureError.invalid("Task change not confirmed. Refresh before retrying. " + error.localizedDescription)
        }
    }
}
