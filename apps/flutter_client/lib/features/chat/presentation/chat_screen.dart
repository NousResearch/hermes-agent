import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../../design/hermes_components.dart';
import '../../../design/hermes_theme.dart';
import '../../connections/data/connection_store.dart';

class ChatScreen extends ConsumerWidget {
  const ChatScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final profile = ref.watch(selectedConnectionProvider);
    final colors = context.hermesColors;
    return HermesShell(
      title: 'Chat',
      subtitle: profile == null ? 'No active instance' : profile.name,
      leading: IconButton(
        tooltip: 'Sessions',
        onPressed: () => context.go('/sessions'),
        icon: const Icon(Icons.arrow_back),
      ),
      children: [
        HermesPanel(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
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
                  const HermesBadge(
                    label: 'prototype',
                    icon: Icons.construction_outlined,
                    tone: HermesBadgeTone.warning,
                  ),
                ],
              ),
              const SizedBox(height: 16),
              _ChatBubble(
                label: 'Hermes Remote',
                text:
                    'This screen is styled like the Desktop shell but intentionally stops before sending prompts. The production chat screen will connect through /api/ws, wait for gateway.ready, then bind a single explicit session id.',
              ),
              const SizedBox(height: 10),
              _ChatBubble(
                label: 'Next contracts',
                text:
                    'Create session, resume session, stream messages, approvals, tool activity, reconnects, and native mobile notification surfaces.',
                accent: true,
              ),
            ],
          ),
        ),
        HermesPanel(
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Icon(Icons.security_outlined, color: colors.warm, size: 18),
              const SizedBox(width: 10),
              Expanded(
                child: Text(
                  'Before chat ships, the UI must always show the active host/profile so prompts cannot accidentally land in the wrong desktop agent.',
                  style: TextStyle(color: colors.secondaryText),
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

class _ChatBubble extends StatelessWidget {
  const _ChatBubble(
      {required this.label, required this.text, this.accent = false});

  final String label;
  final String text;
  final bool accent;

  @override
  Widget build(BuildContext context) {
    final colors = context.hermesColors;
    final color = accent ? colors.accentSoft : colors.card;
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: color,
        borderRadius: BorderRadius.circular(10),
        border: Border.all(
            color: accent ? colors.borderPrimary : colors.borderSecondary),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label, style: Theme.of(context).textTheme.labelMedium),
          const SizedBox(height: 5),
          Text(text, style: TextStyle(color: colors.secondaryText)),
        ],
      ),
    );
  }
}
