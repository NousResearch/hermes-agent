import SwiftUI

/// Routes between the connection wizard and the main app based on `AppState.phase`.
struct RootView: View {
    @EnvironmentObject private var appState: AppState

    var body: some View {
        ZStack {
            Theme.background.ignoresSafeArea()

            switch appState.phase {
            case .needsSetup, .failed:
                ConnectionSetupView()
            case .connecting:
                ConnectingView()
            case .ready:
                SessionListView()
            }
        }
    }
}

private struct ConnectingView: View {
    var body: some View {
        VStack(spacing: 18) {
            HermesMark(size: 64)
            ProgressView()
                .tint(Theme.accent)
            Text("Connecting to Hermes…")
                .font(.subheadline)
                .foregroundStyle(Theme.textSecondary)
        }
    }
}

/// The winged-helmet inspired wordmark used across launch / empty states.
struct HermesMark: View {
    var size: CGFloat = 48

    var body: some View {
        ZStack {
            Circle()
                .fill(Theme.surface)
                .overlay(Circle().stroke(Theme.accent.opacity(0.4), lineWidth: 1))
            Image(systemName: "paperplane.fill")
                .font(.system(size: size * 0.42, weight: .bold))
                .foregroundStyle(Theme.accent)
                .rotationEffect(.degrees(-12))
        }
        .frame(width: size, height: size)
        .shadow(color: Theme.accent.opacity(0.25), radius: 12)
    }
}
