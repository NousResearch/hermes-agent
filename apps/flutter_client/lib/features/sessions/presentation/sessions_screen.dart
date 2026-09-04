import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../../design/hermes_components.dart';
import '../../../design/hermes_theme.dart';
import '../../connections/data/connection_store.dart';

class SessionsScreen extends ConsumerWidget {
  const SessionsScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final profile = ref.watch(selectedConnectionProvider);
    final colors = context.hermesColors;
    return HermesShell(
      title: 'Sessions',
      subtitle: profile == null ? 'Choose an instance first.' : profile.name,
      leading: IconButton(
        tooltip: 'Instances',
        onPressed: () => context.go('/'),
        icon: const Icon(Icons.arrow_back),
      ),
      children: [
        HermesPanel(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  const HermesGlyph(size: 28),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          profile == null
                              ? 'No instance selected'
                              : profile.name,
                          style: Theme.of(context).textTheme.titleMedium,
                        ),
                        if (profile != null)
                          HermesCodeText(profile.baseUrl.toString()),
                      ],
                    ),
                  ),
                  HermesBadge(
                    label: profile == null ? 'inactive' : 'selected',
                    icon: profile == null
                        ? Icons.warning_amber_rounded
                        : Icons.radio_button_checked,
                    tone: profile == null
                        ? HermesBadgeTone.warning
                        : HermesBadgeTone.success,
                  ),
                ],
              ),
              const SizedBox(height: 16),
              FilledButton.icon(
                onPressed: profile == null ? null : () => context.go('/chat'),
                icon: const Icon(Icons.add_comment_outlined, size: 16),
                label: const Text('New Session'),
              ),
              const SizedBox(height: 8),
              OutlinedButton.icon(
                onPressed: profile == null ? null : () => context.go('/chat'),
                icon: const Icon(Icons.history, size: 16),
                label: const Text('Resume Session'),
              ),
            ],
          ),
        ),
        HermesPanel(
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Icon(Icons.account_tree_outlined, color: colors.accent, size: 18),
              const SizedBox(width: 10),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('Clean boundaries stay explicit',
                        style: Theme.of(context).textTheme.titleSmall),
                    const SizedBox(height: 5),
                    Text(
                      'New Session and Resume stay separate so the app never silently crosses conversation context. The next slice replaces this placeholder with the Hermes dashboard session list API.',
                      style: TextStyle(color: colors.secondaryText),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}
