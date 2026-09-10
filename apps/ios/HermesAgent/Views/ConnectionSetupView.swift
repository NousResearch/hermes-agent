import SwiftUI

/// First-run wizard: connect to a self-hosted Hermes Gateway by entering the
/// server URL and API key. Credentials are persisted via `Connection.save()`
/// (key → Keychain, URL → UserDefaults).
struct ConnectionSetupView: View {
    @EnvironmentObject private var appState: AppState

    @State private var baseURL: String = ""
    @State private var apiKey: String = ""
    @State private var showKey = false

    private var failureMessage: String? {
        if case let .failed(message) = appState.phase { return message }
        return nil
    }

    private var canConnect: Bool {
        Connection(baseURL: baseURL.trimmingCharacters(in: .whitespaces), apiKey: apiKey).isValid
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                header

                field(title: "Server URL",
                      hint: "https://hermes.your-domain.com",
                      systemImage: "link") {
                    TextField("https://…", text: $baseURL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .keyboardType(.URL)
                        .textContentType(.URL)
                }

                field(title: "API Key",
                      hint: "Matches API_SERVER_KEY on your gateway",
                      systemImage: "key.fill") {
                    HStack {
                        Group {
                            if showKey {
                                TextField("sk-…", text: $apiKey)
                            } else {
                                SecureField("sk-…", text: $apiKey)
                            }
                        }
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()

                        Button { showKey.toggle() } label: {
                            Image(systemName: showKey ? "eye.slash" : "eye")
                                .foregroundStyle(Theme.textSecondary)
                        }
                    }
                }

                if let failureMessage {
                    Label(failureMessage, systemImage: "exclamationmark.triangle.fill")
                        .font(.footnote)
                        .foregroundStyle(Theme.danger)
                        .padding(12)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Theme.danger.opacity(0.12), in: RoundedRectangle(cornerRadius: 10))
                }

                Button {
                    let conn = Connection(baseURL: baseURL.trimmingCharacters(in: .whitespaces), apiKey: apiKey)
                    Task { await appState.connect(conn) }
                } label: {
                    HStack {
                        if appState.phase == .connecting {
                            ProgressView().tint(.black)
                        }
                        Text("Connect")
                            .fontWeight(.semibold)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 14)
                }
                .background(canConnect ? Theme.accent : Theme.stroke,
                            in: RoundedRectangle(cornerRadius: Theme.corner))
                .foregroundStyle(canConnect ? .black : Theme.textFaint)
                .disabled(!canConnect || appState.phase == .connecting)

                tips
            }
            .padding(24)
        }
        .onAppear {
            baseURL = appState.connection.baseURL
            apiKey = appState.connection.apiKey
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 14) {
            HermesMark(size: 60)
            Text("Hermes in your pocket")
                .font(.largeTitle.bold())
                .foregroundStyle(Theme.textPrimary)
            Text("Connect to your self-hosted Hermes Gateway to manage autonomous tasks, skills, and memory from your iPhone.")
                .font(.subheadline)
                .foregroundStyle(Theme.textSecondary)
        }
        .padding(.top, 20)
    }

    private func field<Content: View>(title: String, hint: String, systemImage: String,
                                      @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Label(title, systemImage: systemImage)
                .font(.subheadline.weight(.medium))
                .foregroundStyle(Theme.textPrimary)
            content()
                .padding(14)
                .background(Theme.surface, in: RoundedRectangle(cornerRadius: Theme.corner))
                .overlay(RoundedRectangle(cornerRadius: Theme.corner).stroke(Theme.stroke, lineWidth: 1))
                .foregroundStyle(Theme.textPrimary)
            Text(hint)
                .font(.caption)
                .foregroundStyle(Theme.textFaint)
        }
    }

    private var tips: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Before you connect")
                .font(.caption.weight(.semibold))
                .foregroundStyle(Theme.textSecondary)
            ForEach([
                "Run the gateway with API_SERVER_ENABLED=true",
                "Expose it securely via Tailscale, Cloudflare Tunnel, or a reverse proxy",
                "Your key is stored only in the iOS Keychain"
            ], id: \.self) { tip in
                Label(tip, systemImage: "checkmark.circle")
                    .font(.caption)
                    .foregroundStyle(Theme.textFaint)
            }
        }
        .padding(.top, 8)
    }
}
