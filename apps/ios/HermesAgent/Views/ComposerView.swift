import SwiftUI

/// Message composer with slash-command autocomplete and a send/stop button.
struct ComposerView: View {
    @ObservedObject var vm: ChatViewModel
    var focused: FocusState<Bool>.Binding

    private var slashMatches: [SlashCommand] {
        SlashCommand.matches(for: vm.draft.trimmingCharacters(in: .whitespaces))
    }

    var body: some View {
        VStack(spacing: 0) {
            if !slashMatches.isEmpty {
                SlashCommandMenu(matches: slashMatches) { command in
                    vm.draft = command.token + " "
                }
            }

            HStack(alignment: .bottom, spacing: 10) {
                TextField("Message Hermes…  (try /skills)", text: $vm.draft, axis: .vertical)
                    .focused(focused)
                    .lineLimit(1...6)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .background(Theme.surface, in: RoundedRectangle(cornerRadius: 20))
                    .overlay(RoundedRectangle(cornerRadius: 20).stroke(Theme.stroke, lineWidth: 1))
                    .foregroundStyle(Theme.textPrimary)

                actionButton
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
        }
        .background(Theme.surfaceRaised)
    }

    @ViewBuilder
    private var actionButton: some View {
        if vm.isStreaming {
            Button { vm.stop() } label: {
                Image(systemName: "stop.fill")
                    .font(.system(size: 18, weight: .bold))
                    .foregroundStyle(.black)
                    .frame(width: 40, height: 40)
                    .background(Theme.danger, in: Circle())
            }
        } else {
            Button { vm.send() } label: {
                Image(systemName: "arrow.up")
                    .font(.system(size: 18, weight: .bold))
                    .foregroundStyle(vm.canSend ? .black : Theme.textFaint)
                    .frame(width: 40, height: 40)
                    .background(vm.canSend ? Theme.accent : Theme.surface, in: Circle())
            }
            .disabled(!vm.canSend)
        }
    }
}

/// Inline autocomplete list shown above the composer while typing `/…`.
struct SlashCommandMenu: View {
    let matches: [SlashCommand]
    let onSelect: (SlashCommand) -> Void

    var body: some View {
        ScrollView {
            VStack(spacing: 0) {
                ForEach(matches) { command in
                    Button { onSelect(command) } label: {
                        HStack(spacing: 10) {
                            Text(command.token)
                                .font(.system(.subheadline, design: .monospaced).weight(.semibold))
                                .foregroundStyle(Theme.accent)
                            VStack(alignment: .leading, spacing: 1) {
                                Text(command.summary)
                                    .font(.footnote)
                                    .foregroundStyle(Theme.textPrimary)
                                if let usage = command.usage {
                                    Text(usage)
                                        .font(.caption2.monospaced())
                                        .foregroundStyle(Theme.textFaint)
                                }
                            }
                            Spacer()
                        }
                        .padding(.horizontal, 16)
                        .padding(.vertical, 9)
                        .contentShape(Rectangle())
                    }
                    if command.id != matches.last?.id {
                        Divider().overlay(Theme.stroke)
                    }
                }
            }
        }
        .frame(maxHeight: 200)
        .background(Theme.surface)
        .overlay(Rectangle().frame(height: 1).foregroundStyle(Theme.stroke), alignment: .top)
    }
}
