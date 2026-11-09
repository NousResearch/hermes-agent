import SwiftUI

/// Reusable empty / zero-data state with an optional async action.
struct EmptyStateView: View {
    let icon: String
    let title: String
    let message: String
    var actionTitle: String?
    var action: (() async -> Void)?

    @State private var isRunning = false

    var body: some View {
        VStack(spacing: 14) {
            Image(systemName: icon)
                .font(.system(size: 42, weight: .light))
                .foregroundStyle(Theme.textFaint)
            Text(title)
                .font(.title3.weight(.semibold))
                .foregroundStyle(Theme.textPrimary)
            Text(message)
                .font(.subheadline)
                .foregroundStyle(Theme.textSecondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)

            if let actionTitle, let action {
                Button {
                    Task { isRunning = true; await action(); isRunning = false }
                } label: {
                    HStack {
                        if isRunning { ProgressView().tint(.black) }
                        Text(actionTitle).fontWeight(.semibold)
                    }
                    .padding(.horizontal, 22).padding(.vertical, 11)
                    .background(Theme.accent, in: Capsule())
                    .foregroundStyle(.black)
                }
                .padding(.top, 6)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
