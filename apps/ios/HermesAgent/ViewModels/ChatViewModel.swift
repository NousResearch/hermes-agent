import SwiftUI

/// Drives a single chat session: loads history, streams agent turns over SSE,
/// surfaces live tool-progress, and routes slash commands to the TUI gateway.
///
/// Two-phase lifecycle: constructed with only the `session` (so it can back a
/// `@StateObject`), then `bind(appState:)` wires up the networking clients once
/// the SwiftUI environment is available.
@MainActor
final class ChatViewModel: ObservableObject {
    @Published var messages: [ChatMessage] = []
    @Published var draft: String = ""
    @Published var liveTools: [ToolEvent] = []
    @Published var reasoning: String = ""
    @Published var isStreaming = false
    @Published var isLoading = false
    @Published var errorMessage: String?
    @Published var statusLine: String?

    let session: HermesSession

    private var api: HermesAPIClient?
    private var gateway: TUIGatewayClient?
    private var sse: SSEClient?
    private var model: String?
    private var streamTask: Task<Void, Never>?
    private var isBound = false

    init(session: HermesSession) {
        self.session = session
    }

    /// Wire up networking from the live AppState. Idempotent.
    func bind(appState: AppState) {
        guard !isBound else { return }
        isBound = true
        api = appState.api
        gateway = appState.gateway
        sse = SSEClient(connection: appState.connection)
        model = session.model ?? appState.selectedModel
    }

    // MARK: - History

    func loadHistory() async {
        guard let api else { return }
        isLoading = true
        defer { isLoading = false }
        do {
            let history = try await api.messages(sessionId: session.id)
            messages = history.filter { $0.role != .system }
        } catch {
            errorMessage = (error as? HermesError)?.errorDescription ?? error.localizedDescription
        }
    }

    // MARK: - Sending

    var canSend: Bool {
        !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && !isStreaming
    }

    func send() {
        let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !isStreaming else { return }
        draft = ""

        if SlashCommand.isSlashCommand(text) {
            Task { await dispatchSlash(text) }
            return
        }

        messages.append(ChatMessage(role: .user, content: text, sessionId: session.id))
        startStream(message: text)
    }

    private func startStream(message: String) {
        guard let sse else { return }
        isStreaming = true
        liveTools = []
        reasoning = ""
        statusLine = "Hermes is thinking…"

        let draftId = "draft-" + UUID().uuidString
        messages.append(ChatMessage(id: draftId, role: .assistant, content: "", sessionId: session.id))

        streamTask = Task {
            do {
                for try await event in sse.streamChat(sessionId: session.id, message: message, model: model) {
                    apply(event, draftId: draftId)
                }
            } catch {
                errorMessage = (error as? HermesError)?.errorDescription ?? error.localizedDescription
            }
            finishStream(draftId: draftId)
        }
    }

    private func apply(_ event: StreamEvent, draftId: String) {
        switch event {
        case .runStarted:
            statusLine = "Run started"
        case .messageStarted:
            statusLine = "Responding…"
        case .assistantDelta(_, let text):
            appendToDraft(draftId, text)
        case .reasoningDelta(let text):
            reasoning += text
            statusLine = "Reasoning…"
        case .toolStarted(let id, let tool, let preview):
            statusLine = "Running \(ToolEvent.prettify(tool))…"
            upsertTool(ToolEvent(id: id, toolName: tool, label: nil, emoji: nil, preview: preview, status: .running))
        case .toolCompleted(let id, let tool, let preview):
            upsertTool(ToolEvent(id: id, toolName: tool, label: nil, emoji: nil, preview: preview, status: .completed))
        case .toolFailed(let id, let tool, let preview):
            upsertTool(ToolEvent(id: id, toolName: tool, label: nil, emoji: nil, preview: preview, status: .failed))
        case .assistantCompleted(_, let content):
            if let content, !content.isEmpty { setDraft(draftId, content) }
        case .runCompleted(let usage):
            if let usage, let out = usage.outputTokens {
                statusLine = "Done · \(out) tokens"
            }
        case .approvalRequest(_, let choices):
            statusLine = "Approval needed: \(choices.joined(separator: " / "))"
        case .error(let message):
            errorMessage = message
        case .done, .unknown:
            break
        }
    }

    private func finishStream(draftId: String) {
        isStreaming = false
        statusLine = nil
        if let idx = messages.firstIndex(where: { $0.id == draftId }) {
            messages[idx].toolEvents = liveTools
            if messages[idx].content.isEmpty && liveTools.isEmpty {
                messages.remove(at: idx)
            }
        }
        liveTools = []
        reasoning = ""
    }

    func stop() {
        streamTask?.cancel()
        streamTask = nil
        isStreaming = false
        statusLine = nil
    }

    // MARK: - Slash commands

    private func dispatchSlash(_ command: String) async {
        guard let gateway else { return }
        statusLine = "Dispatching \(command)…"
        do {
            let result = try await gateway.dispatch(command: command, sessionId: session.id)
            let output = (result["output"] as? String)
                ?? (result["message"] as? String)
                ?? "Command \(command) dispatched."
            messages.append(ChatMessage(role: .system, content: output, sessionId: session.id))
        } catch {
            errorMessage = (error as? HermesError)?.errorDescription ?? error.localizedDescription
        }
        statusLine = nil
    }

    // MARK: - Mutation helpers

    private func appendToDraft(_ id: String, _ text: String) {
        guard let idx = messages.firstIndex(where: { $0.id == id }) else { return }
        messages[idx].content += text
    }

    private func setDraft(_ id: String, _ text: String) {
        guard let idx = messages.firstIndex(where: { $0.id == id }) else { return }
        messages[idx].content = text
    }

    private func upsertTool(_ event: ToolEvent) {
        if let idx = liveTools.firstIndex(where: { $0.id == event.id }) {
            liveTools[idx].status = event.status
            if let p = event.preview { liveTools[idx].preview = p }
        } else {
            liveTools.append(event)
        }
    }
}
