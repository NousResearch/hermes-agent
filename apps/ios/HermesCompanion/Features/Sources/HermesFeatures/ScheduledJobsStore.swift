import Foundation
import Observation
import HermesCore

public enum ScheduledJobAction: String, CaseIterable, Identifiable {
    case pause, resume, runNow, delete
    public var id: String { rawValue }
    public var title: String {
        switch self { case .pause: "Pause"; case .resume: "Resume"; case .runNow: "Run now"; case .delete: "Delete" }
    }
    var suffix: String {
        switch self { case .pause: "/pause"; case .resume: "/resume"; case .runNow: "/trigger"; case .delete: "" }
    }
}

public struct ScheduledJobDraft {
    public var name: String
    public var schedule: String
    public var prompt: String
    public var model: String
    public var deliver: String
    public init(job: JSON = .null) {
        name = job["name"].string ?? ""
        schedule = job["schedule"]["expr"].string ?? job["schedule_display"].string
            ?? job["schedule"]["display"].string ?? job["schedule"].string ?? ""
        prompt = job["prompt"].string ?? ""
        model = job["model"].string ?? ""
        deliver = job["deliver"].string ?? "local"
    }
    public var body: JSON {
        .object(["name": .string(name), "schedule": .string(schedule), "prompt": .string(prompt),
                 "model": .string(model), "deliver": .string(deliver)])
    }
    public func changes(from job: JSON) -> JSON {
        let original = ScheduledJobDraft(job: job).body
        return .object((body.object ?? [:]).filter { original[$0.key] != $0.value })
    }
}

/// Every write is bound to a connection generation and the concrete (profile, job ID).
@MainActor @Observable public final class ScheduledJobsStore {
    public private(set) var jobs: [JSON] = []
    public private(set) var runs: [JSON] = []
    public private(set) var selectedJob: JSON?
    public private(set) var loading = false
    public private(set) var runsLoading = false
    public private(set) var mutating = false
    public private(set) var needsRefresh = true
    public private(set) var contextID = UUID()
    public var errorMessage: String?
    public var unknownOutcomeMessage: String?
    public var canMutate: Bool { connected && !loading && !mutating && !needsRefresh }
    private let transport: CompanionTransport
    private var connected = false
    private var profile = ""
    private var runVersion = UUID()

    public init(transport: CompanionTransport) { self.transport = transport }

    public func configure(connected: Bool, profile: String) {
        guard self.connected != connected || self.profile != profile else { return }
        contextID = UUID(); runVersion = UUID()
        self.connected = connected; self.profile = profile
        jobs = []; runs = []; selectedJob = nil
        loading = false; runsLoading = false; mutating = false; needsRefresh = true
        errorMessage = nil; unknownOutcomeMessage = nil
    }

    public func refresh() async {
        guard connected, !loading, !mutating else { return }
        let token = contextID
        loading = true; needsRefresh = true; errorMessage = nil
        do {
            let value = try await transport.http("/api/cron/jobs", method: "GET", body: nil,
                query: ["profile": profile.isEmpty ? "all" : profile])
            guard token == contextID else { return }
            guard let rows = value.array else { throw FeatureError.invalid("Invalid scheduled-job list from Hermes.") }
            jobs = rows
            if let previous = selectedJob {
                selectedJob = rows.first { sameJob($0, previous) }
                if selectedJob == nil { runs = []; runVersion = UUID(); runsLoading = false }
            }
            loading = false; needsRefresh = false; unknownOutcomeMessage = nil
        } catch {
            if token == contextID { loading = false; needsRefresh = true; errorMessage = error.localizedDescription }
        }
    }

    public func select(_ job: JSON) async {
        guard connected, !needsRefresh, let current = jobs.first(where: { sameJob($0, job) }),
              let (id, concrete) = identity(current) else { return }
        selectedJob = current; runs = []; runsLoading = true; runVersion = UUID()
        let token = runVersion; let connection = contextID
        do {
            let result = try await transport.http("/api/cron/jobs/\(id)/runs", method: "GET", body: nil,
                query: ["profile": concrete, "limit": "20"])
            guard token == runVersion, connection == contextID else { return }
            guard let rows = result["runs"].array else { throw FeatureError.invalid("Invalid scheduled-job history from Hermes.") }
            runs = rows; runsLoading = false
        } catch {
            if token == runVersion, connection == contextID { runsLoading = false; errorMessage = error.localizedDescription }
        }
    }

    public func create(_ body: JSON, concreteProfile: String, context: UUID) async -> Bool {
        guard context == contextID, canMutate else { return false }
        guard !concreteProfile.isEmpty, concreteProfile != "all",
              !(body["prompt"].string ?? "").trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              !(body["schedule"].string ?? "").trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            errorMessage = "Choose one profile, a schedule, and a prompt."; return false
        }
        return await mutate(path: "/api/cron/jobs", method: "POST", body: body,
                            profile: concreteProfile, expectedID: nil, context: context)
    }

    public func update(_ job: JSON, changed: JSON, context: UUID) async -> Bool {
        guard let (id, concrete) = writableIdentity(job, context: context),
              let fields = changed.object, !fields.isEmpty,
              Set(fields.keys).isSubset(of: ["name", "schedule", "prompt", "model", "deliver"]) else { return false }
        return await mutate(path: "/api/cron/jobs/\(id)", method: "PUT", body: .object(["updates": changed]),
                            profile: concrete, expectedID: job["id"].string, context: context)
    }

    public func perform(_ action: ScheduledJobAction, job: JSON, context: UUID) async -> Bool {
        guard let (id, concrete) = writableIdentity(job, context: context) else { return false }
        return await mutate(path: "/api/cron/jobs/\(id)\(action.suffix)",
                            method: action == .delete ? "DELETE" : "POST", body: nil,
                            profile: concrete, expectedID: job["id"].string, context: context)
    }

    private func writableIdentity(_ job: JSON, context: UUID) -> (String, String)? {
        guard context == contextID, canMutate,
              let current = jobs.first(where: { sameJob($0, job) }) else { return nil }
        return identity(current)
    }

    private func identity(_ job: JSON) -> (String, String)? {
        guard let raw = job["id"].string, let id = try? WorkspaceLibrary.pathID(raw),
              let concrete = job["profile"].string, !concrete.isEmpty, concrete != "all" else { return nil }
        return (id, concrete)
    }

    private func sameJob(_ lhs: JSON, _ rhs: JSON) -> Bool {
        lhs["id"].string != nil && lhs["id"] == rhs["id"] && lhs["profile"] == rhs["profile"]
    }

    private func mutate(path: String, method: String, body: JSON?, profile: String,
                        expectedID: String?, context: UUID) async -> Bool {
        guard context == contextID, canMutate else { return false }
        mutating = true; errorMessage = nil; unknownOutcomeMessage = nil
        do {
            let receipt = try await transport.http(path, method: method, body: body, query: ["profile": profile])
            guard context == contextID else { return false }
            if method == "DELETE" {
                guard receipt["ok"].bool == true else { throw FeatureError.invalid("Deletion was not confirmed.") }
            } else {
                guard let id = receipt["id"].string, !id.isEmpty,
                      expectedID == nil || id == expectedID,
                      receipt["profile"].string == nil || receipt["profile"].string == profile else {
                    throw FeatureError.invalid("Hermes returned an unconfirmed scheduled-job receipt.")
                }
            }
            mutating = false; needsRefresh = true
            await refresh()
            return context == contextID && !needsRefresh
        } catch {
            guard context == contextID else { return false }
            mutating = false; needsRefresh = true
            errorMessage = error.localizedDescription
            unknownOutcomeMessage = "Outcome is unknown. Refresh and check the job before repeating this action; your Mac may still be working."
            return false
        }
    }

    #if DEBUG
    public func installPreview(_ rows: [JSON]) {
        guard !connected else { return }
        jobs = rows
    }
    #endif
}
