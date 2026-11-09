import Foundation

/// A single tool-execution step surfaced by the agent while a turn streams.
/// Backed by the `tool.started` / `tool.completed` / `tool.progress` SSE events
/// (and the `hermes.tool.progress` event on the OpenAI-compatible stream).
struct ToolEvent: Identifiable, Hashable {
    enum Status: String {
        case running
        case completed
        case failed
    }

    let id: String          // tool_call_id when available, else a generated id
    var toolName: String
    var label: String?      // human-readable label, e.g. "Searching web…"
    var emoji: String?      // optional glyph the backend supplies
    var preview: String?    // short arg/result preview
    var status: Status

    /// Reasoning / "thinking" deltas arrive with tool_name == "_thinking".
    var isThinking: Bool { toolName == "_thinking" }

    /// Friendly title for the chip.
    var title: String {
        if let label, !label.isEmpty { return label }
        if isThinking { return "Thinking…" }
        return Self.prettify(toolName)
    }

    var symbol: String {
        if let emoji, !emoji.isEmpty { return emoji }
        if isThinking { return "🧠" }
        switch toolName {
        case let t where t.contains("web") || t.contains("search"): return "🔍"
        case let t where t.contains("terminal") || t.contains("shell") || t.contains("bash"): return "⌨️"
        case let t where t.contains("file") || t.contains("read") || t.contains("write"): return "📄"
        case let t where t.contains("browser"): return "🌐"
        default: return "🛠️"
        }
    }

    static func prettify(_ raw: String) -> String {
        raw.replacingOccurrences(of: "_", with: " ")
            .replacingOccurrences(of: ".", with: " ")
            .capitalized
    }
}
