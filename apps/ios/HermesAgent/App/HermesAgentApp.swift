import SwiftUI

@main
struct HermesAgentApp: App {
    @StateObject private var appState = AppState()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(appState)
                .tint(Theme.accent)
                .preferredColorScheme(.dark)
                .task { await appState.bootstrap() }
        }
    }
}
