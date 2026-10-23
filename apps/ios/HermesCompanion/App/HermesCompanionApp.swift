import SwiftUI
import HermesCore
import HermesFeatures


@main
struct HermesCompanionApp: App {
    @State private var store = CompanionStore(transport: Gateway())
    @Environment(\.scenePhase) private var scenePhase
    var body: some Scene {
        WindowGroup {
            Group {
                if store.isConnected || !store.host.isEmpty || store.isPreview {
                    MobileWorkspaceView(store: store)
                } else { ConnectView(store: store) }
            }
            .tint(HermesTheme.accent)
            .task {
                #if DEBUG
                if ProcessInfo.processInfo.arguments.contains("--ui-preview") {
                    PreviewContent.install(in: store)
                }
                #endif
            }
            .onChange(of: scenePhase) { _, phase in
                guard !store.isPreview else { return }
                if phase == .background { store.suspend() }
                else if phase == .active, !store.host.isEmpty, !store.isConnected {
                    Task { await store.reconnect() }
                }
            }
        }
    }
}

#if DEBUG
@MainActor enum PreviewContent {
    static func install(in store: CompanionStore) {
        store.beginPreview(); store.host = "Preview Mac"; store.selectedProfile = "Hermes"
        store.profiles = [.object(["name": .string("Hermes"), "description": .string("Your personal agent")])]
        store.sessions = [.object(["id": .string("preview"), "title": .string("A little closer to home"), "preview": .string("Hermes, from your phone.")])]
        var state = SessionState(sessionID: "preview", storedID: "preview")
        state.messages = [
            ChatMessage(role: "user", text: "Can we pick up where we left off on the Mac?"),
            ChatMessage(role: "assistant", text: "Right here. Your conversations, agents, and Kanban stay on your Mac. This is your phone's window into Hermes."),
            ChatMessage(role: "user", text: "And I can control the work from here?"),
            ChatMessage(role: "assistant", text: "Send a message, stop a running turn, manage a task, or respond to an approval. Each action still goes through the Mac's existing checks.")
        ]
        state.activity = "Sample conversation"
        store.session = state; store.currentModel = "Profile default"
        store.selectedBoard = "preview"
        store.boards = [.object(["slug": .string("preview"), "name": .string("Companion")])]
        store.columns = [
            .object(["name": .string("triage"), "tasks": .array([.object(["id": .string("preview-task"), "title": .string("Check gateway connection"), "body": .string("Sample task. No connection or live task has been created."), "status": .string("triage")])])]),
            .object(["name": .string("running"), "tasks": .array([])]),
            .object(["name": .string("done"), "tasks": .array([])])
        ]
        store.scheduledJobs.installPreview([
            .object(["id": .string("preview-daily"), "profile": .string("Hermes"), "name": .string("Morning workspace review"),
                     "schedule": .object(["expr": .string("0 9 * * 1-5")]), "enabled": .bool(true),
                     "prompt": .string("Sample job: review the board and summarize work that needs attention.")]),
            .object(["id": .string("preview-weekly"), "profile": .string("Hermes"), "name": .string("Weekly maintenance"),
                     "schedule": .object(["expr": .string("0 17 * * 5")]), "state": .string("paused"),
                     "prompt": .string("Sample job: check upgrades and prepare a maintenance summary.")])
        ])
        store.library.installPreview(.usage, value: .object([
            "totals": .object(["total_sessions": .number(42), "total_api_calls": .number(186),
                               "total_input": .number(142000), "total_output": .number(28000)]),
            "daily": .array([5, 8, 3, 9, 6, 4, 7].enumerated().map { index, count in
                .object(["day": .string("Day \(index + 1)"), "sessions": .number(Double(count))])
            }),
            "by_model": .array([])
        ]))
    }
}
#endif
