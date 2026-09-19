import Foundation

/// Client-side catalog of slash commands surfaced in the composer autocomplete.
/// When the user submits one, it is forwarded to the TUI Gateway's
/// `command.dispatch` JSON-RPC method (see `TUIGatewayClient`).
struct SlashCommand: Identifiable, Hashable {
    let name: String          // without leading slash, e.g. "model"
    let summary: String
    let usage: String?

    var id: String { name }
    var token: String { "/" + name }

    static let catalog: [SlashCommand] = [
        SlashCommand(name: "new", summary: "Start a fresh session", usage: "/new"),
        SlashCommand(name: "model", summary: "Switch the active model", usage: "/model <name>"),
        SlashCommand(name: "models", summary: "List available models", usage: "/models"),
        SlashCommand(name: "skills", summary: "Manage agent skills", usage: "/skills"),
        SlashCommand(name: "personality", summary: "Adjust agent personality", usage: "/personality"),
        SlashCommand(name: "memory", summary: "View long-term memory", usage: "/memory"),
        SlashCommand(name: "compress", summary: "Summarize & compress the session", usage: "/compress"),
        SlashCommand(name: "tools", summary: "List available tools", usage: "/tools"),
        SlashCommand(name: "title", summary: "Rename this session", usage: "/title <text>")
    ]

    /// Returns matching commands for a partial token like "/mod".
    static func matches(for input: String) -> [SlashCommand] {
        guard input.hasPrefix("/") else { return [] }
        let query = input.dropFirst().lowercased()
        if query.isEmpty { return catalog }
        // Only autocomplete while typing the command word (no spaces yet).
        guard !query.contains(" ") else { return [] }
        return catalog.filter { $0.name.hasPrefix(query) }
    }

    static func isSlashCommand(_ text: String) -> Bool {
        text.trimmingCharacters(in: .whitespaces).hasPrefix("/")
    }
}
