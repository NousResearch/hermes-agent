import SwiftUI

/// Model picker backed by `GET /v1/models`. The selection maps to the Hermes
/// `runtime_provider` model id sent on each chat turn.
struct ModelPickerView: View {
    @EnvironmentObject private var appState: AppState
    @Environment(\.dismiss) private var dismiss
    @State private var query = ""

    private var filtered: [ModelInfo] {
        guard !query.isEmpty else { return appState.availableModels }
        return appState.availableModels.filter { $0.id.localizedCaseInsensitiveContains(query) }
    }

    var body: some View {
        NavigationStack {
            ZStack {
                Theme.background.ignoresSafeArea()
                Group {
                    if appState.availableModels.isEmpty {
                        EmptyStateView(icon: "cpu",
                                       title: "No models loaded",
                                       message: "Pull the model list from your gateway.",
                                       actionTitle: "Reload") { await appState.loadModels() }
                    } else {
                        List {
                            ForEach(filtered) { model in
                                Button {
                                    appState.selectedModel = model.id
                                    dismiss()
                                } label: {
                                    HStack {
                                        VStack(alignment: .leading, spacing: 2) {
                                            Text(model.id)
                                                .foregroundStyle(Theme.textPrimary)
                                            if let owner = model.ownedBy {
                                                Text(owner).font(.caption).foregroundStyle(Theme.textFaint)
                                            }
                                        }
                                        Spacer()
                                        if appState.selectedModel == model.id {
                                            Image(systemName: "checkmark")
                                                .foregroundStyle(Theme.accent)
                                        }
                                    }
                                }
                                .listRowBackground(Theme.surface)
                            }
                        }
                        .listStyle(.plain)
                        .scrollContentBackground(.hidden)
                        .searchable(text: $query, prompt: "Search models")
                    }
                }
            }
            .navigationTitle("Model")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
            .task { if appState.availableModels.isEmpty { await appState.loadModels() } }
        }
    }
}
