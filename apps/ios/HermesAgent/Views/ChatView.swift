import SwiftUI

/// The core conversation screen: streaming transcript, live tool-progress bar,
/// reasoning disclosure, and a composer with slash-command autocomplete.
struct ChatView: View {
    @EnvironmentObject private var appState: AppState
    @StateObject private var vm: ChatViewModel
    @FocusState private var composerFocused: Bool

    init(session: HermesSession) {
        _vm = StateObject(wrappedValue: ChatViewModel(session: session))
    }

    var body: some View {
        VStack(spacing: 0) {
            transcript
            footer
        }
        .background(Theme.background)
        .navigationTitle(vm.session.displayTitle)
        .navigationBarTitleDisplayMode(.inline)
        .toolbarBackground(Theme.background, for: .navigationBar)
        .task {
            vm.bind(appState: appState)
            await vm.loadHistory()
        }
    }

    private var transcript: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 14) {
                    if vm.isLoading {
                        ProgressView().tint(Theme.accent).frame(maxWidth: .infinity).padding()
                    }
                    ForEach(vm.messages) { message in
                        MessageBubbleView(message: message,
                                          isStreaming: vm.isStreaming && message.id == vm.messages.last?.id)
                            .id(message.id)
                    }
                    if !vm.reasoning.isEmpty {
                        ReasoningView(text: vm.reasoning).id("reasoning")
                    }
                    Color.clear.frame(height: 8).id("bottom")
                }
                .padding(.horizontal, 14)
                .padding(.top, 12)
            }
            .scrollDismissesKeyboard(.interactively)
            .onChange(of: vm.messages.last?.content) { _, _ in
                withAnimation(.easeOut(duration: 0.15)) { proxy.scrollTo("bottom", anchor: .bottom) }
            }
            .onChange(of: vm.reasoning) { _, _ in
                proxy.scrollTo("bottom", anchor: .bottom)
            }
        }
    }

    private var footer: some View {
        VStack(spacing: 0) {
            if let error = vm.errorMessage {
                Text(error)
                    .font(.footnote)
                    .foregroundStyle(Theme.danger)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 16).padding(.vertical, 6)
                    .background(Theme.danger.opacity(0.1))
            }

            ToolProgressBar(tools: vm.liveTools, status: vm.statusLine, isStreaming: vm.isStreaming)

            ComposerView(vm: vm, focused: $composerFocused)
        }
    }
}
