import SwiftUI
import HermesFeatures

struct ConnectView: View {
    @Bindable var store: CompanionStore
    @State private var url = ""
    @State private var username = ""
    @State private var password = ""
    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 28) {
                    HStack(spacing: 12) { HermesMark(size: 44); Text("Hermes").font(.title.weight(.semibold)) }
                    VStack(alignment: .leading, spacing: 10) {
                        Text("Your Mac.\nIn your pocket.").font(.largeTitle.weight(.semibold))
                        Text("Connect privately through Tailscale to keep the conversation—and the work—moving.")
                            .foregroundStyle(.secondary)
                    }
                    VStack(alignment: .leading, spacing: 18) {
                        field("MAC ADDRESS") {
                            TextField("https://your-mac.your-tailnet.ts.net", text: $url)
                                .keyboardType(.URL).textInputAutocapitalization(.never).autocorrectionDisabled()
                                .accessibilityLabel("Tailnet URL")
                        }
                        field("HERMES USERNAME") {
                            TextField("Username", text: $username).textContentType(.username)
                                .textInputAutocapitalization(.never).autocorrectionDisabled()
                        }
                        field("PASSWORD") {
                            SecureField("Password", text: $password).textContentType(.password)
                        }
                    }
                    if let error = store.errorMessage {
                        Label(error, systemImage: "exclamationmark.circle").font(.callout).foregroundStyle(.red)
                    }
                    Button {
                        let secret = password; password = ""
                        Task { await store.connect(url: url, username: username, password: secret) }
                    } label: {
                        HStack { if store.connecting { ProgressView().tint(.white) }; Text(store.connecting ? "Connecting" : "Connect to Mac").fontWeight(.semibold); Spacer(); Image(systemName: "arrow.right") }
                            .padding(16).foregroundStyle(.white).background(HermesTheme.accent).clipShape(RoundedRectangle(cornerRadius: 10))
                    }
                    .disabled(store.connecting || url.isEmpty || username.isEmpty || password.isEmpty)
                    .opacity(url.isEmpty || username.isEmpty || password.isEmpty ? 0.5 : 1)
                    #if DEBUG
                    Button {
                        PreviewContent.install(in: store)
                    } label: {
                        Label("Preview interface", systemImage: "eye")
                            .frame(maxWidth: .infinity).padding(.vertical, 8)
                    }
                    .accessibilityIdentifier("preview-interface")
                    Text("Explore sample screens without signing in. No commands are sent.")
                        .font(.footnote).foregroundStyle(.secondary)
                    #endif
                    VStack(alignment: .leading, spacing: 10) {
                        Label("Private connection. Mac-owned execution.", systemImage: "lock.shield")
                        Text("Messages and controls can run tools on your Mac. Existing permissions still apply. Tailscale and Hermes sign-in must both be configured.")
                        Text("Your password is not saved. A connected Mac must stay awake and running Hermes.")
                    }.font(.footnote).foregroundStyle(.secondary)
                }.padding(28)
            }.background(HermesTheme.background)
        }
    }
    private func field<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 9) {
            Text(title).font(.caption2.weight(.semibold)).tracking(1.1).foregroundStyle(.secondary)
            content().padding(.vertical, 10)
            Rectangle().fill(HermesTheme.hairline).frame(height: 1)
        }
    }
}
