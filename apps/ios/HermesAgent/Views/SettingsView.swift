import SwiftUI

struct SettingsView: View {
    @EnvironmentObject private var appState: AppState
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ZStack {
                Theme.background.ignoresSafeArea()
                List {
                    Section("Connection") {
                        row(label: "Server", value: appState.connection.baseURL)
                        row(label: "API Key", value: appState.connection.apiKey.isEmpty ? "—" : "•••• stored in Keychain")
                        row(label: "Status", value: statusText)
                    }
                    .listRowBackground(Theme.surface)

                    Section("Agent") {
                        row(label: "Model", value: appState.selectedModel ?? "—")
                        row(label: "Models available", value: "\(appState.availableModels.count)")
                    }
                    .listRowBackground(Theme.surface)

                    Section {
                        Button(role: .destructive) {
                            appState.signOut()
                            dismiss()
                        } label: {
                            Label("Disconnect", systemImage: "rectangle.portrait.and.arrow.right")
                        }
                    }
                    .listRowBackground(Theme.surface)

                    Section {
                        Text("Hermes Agent for iOS — a native client for your self-hosted Hermes Gateway. Chat, tools, skills and memory, in your pocket.")
                            .font(.caption)
                            .foregroundStyle(Theme.textFaint)
                    }
                    .listRowBackground(Theme.background)
                }
                .listStyle(.insetGrouped)
                .scrollContentBackground(.hidden)
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    private var statusText: String {
        switch appState.phase {
        case .ready: return "Connected"
        case .connecting: return "Connecting…"
        case .failed(let m): return m
        case .needsSetup: return "Not configured"
        }
    }

    private func row(label: String, value: String) -> some View {
        HStack {
            Text(label).foregroundStyle(Theme.textSecondary)
            Spacer()
            Text(value)
                .foregroundStyle(Theme.textPrimary)
                .lineLimit(1)
                .truncationMode(.middle)
        }
        .font(.subheadline)
    }
}
