import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../../design/hermes_components.dart';
import '../../../design/hermes_theme.dart';
import '../../../hermes_api/connection_profile.dart';
import '../../../hermes_api/hermes_rest_client.dart';
import '../../connections/data/connection_store.dart';
import '../data/instance_probe.dart';

class InstanceListScreen extends ConsumerWidget {
  const InstanceListScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final connections = ref.watch(connectionStoreProvider);
    final selected = ref.watch(selectedConnectionProvider);

    return HermesShell(
      title: 'Hermes Remote',
      subtitle:
          'Drive clean Desktop sessions from mobile, tablet, or another desktop.',
      actions: [
        IconButton.filledTonal(
          tooltip: 'Add connection',
          onPressed: () => context.go('/connections/new'),
          icon: const Icon(Icons.add_link),
        ),
      ],
      children: [
        HermesPanel(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Expanded(
                    child: Text(
                      'Instances',
                      style: Theme.of(context).textTheme.titleMedium,
                    ),
                  ),
                  HermesBadge(
                    label: '${connections.length} saved',
                    icon: Icons.dns_outlined,
                  ),
                ],
              ),
              const SizedBox(height: 10),
              for (final profile in connections) ...[
                _InstanceRow(
                  profile: profile,
                  selected: selected?.id == profile.id,
                  onSelect: () {
                    ref.read(selectedConnectionIdProvider.notifier).state =
                        profile.id;
                  },
                  onUse: () {
                    ref.read(selectedConnectionIdProvider.notifier).state =
                        profile.id;
                    context.go('/sessions');
                  },
                ),
                if (profile != connections.last) const SizedBox(height: 8),
              ],
            ],
          ),
        ),
        if (selected != null) _ProbePanel(profile: selected),
        _ArchitectureNote(),
      ],
    );
  }
}

class _InstanceRow extends StatelessWidget {
  const _InstanceRow({
    required this.profile,
    required this.selected,
    required this.onSelect,
    required this.onUse,
  });

  final ConnectionProfile profile;
  final bool selected;
  final VoidCallback onSelect;
  final VoidCallback onUse;

  @override
  Widget build(BuildContext context) {
    final colors = context.hermesColors;
    return InkWell(
      borderRadius: BorderRadius.circular(9),
      onTap: onSelect,
      child: Container(
        padding: const EdgeInsets.all(10),
        decoration: BoxDecoration(
          color: selected ? colors.rowActive : Colors.transparent,
          borderRadius: BorderRadius.circular(9),
          border: Border.all(
              color: selected ? colors.borderPrimary : colors.borderSecondary),
        ),
        child: Row(
          children: [
            Container(
              width: 28,
              height: 28,
              decoration: BoxDecoration(
                color: selected ? colors.accentSoft : colors.rowHover,
                borderRadius: BorderRadius.circular(8),
              ),
              child:
                  Icon(Icons.computer_outlined, color: colors.accent, size: 16),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(profile.name,
                      style: Theme.of(context).textTheme.titleSmall),
                  const SizedBox(height: 3),
                  HermesCodeText(profile.baseUrl.toString()),
                ],
              ),
            ),
            const SizedBox(width: 8),
            FilledButton.icon(
              onPressed: onUse,
              icon: const Icon(Icons.login, size: 15),
              label: const Text('Use'),
            ),
          ],
        ),
      ),
    );
  }
}

class _ProbePanel extends ConsumerWidget {
  const _ProbePanel({required this.profile});

  final ConnectionProfile profile;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final probe = ref.watch(instanceProbeProvider(profile));
    return HermesPanel(
      child: probe.when(
        loading: () => const Row(
          children: [
            SizedBox(
              width: 18,
              height: 18,
              child: CircularProgressIndicator(strokeWidth: 2),
            ),
            SizedBox(width: 12),
            Text('Testing status and mobile bootstrap...'),
          ],
        ),
        error: (error, stackTrace) => _StatusText(
          icon: Icons.error_outline,
          tone: HermesBadgeTone.danger,
          title: 'Status check failed',
          body: '$error',
        ),
        data: (result) => Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _StatusText(
              icon: Icons.check_circle_outline,
              tone: HermesBadgeTone.success,
              title: 'Status reachable',
              body:
                  'Gateway: ${result.status.gatewayState ?? 'unknown'} | Active sessions: ${result.status.activeSessions ?? 0}',
            ),
            const SizedBox(height: 12),
            _StatusText(
              icon: result.bootstrap == null
                  ? Icons.info_outline
                  : Icons.mobile_friendly,
              tone: result.bootstrap == null
                  ? HermesBadgeTone.warning
                  : HermesBadgeTone.neutral,
              title: result.bootstrap == null
                  ? 'Bootstrap endpoint pending'
                  : 'Bootstrap reachable',
              body: result.bootstrap == null
                  ? 'GET /api/mobile/bootstrap is not available yet or requires backend work.'
                  : _bootstrapSummary(result.bootstrap!),
            ),
          ],
        ),
      ),
    );
  }

  String _bootstrapSummary(HermesMobileBootstrap bootstrap) {
    final raw = bootstrap.raw;
    final version = raw['server_version'] ?? 'unknown';
    final authRequired =
        raw['auth_required'] == true ? 'auth required' : 'loopback/no auth';
    final features = raw['features'] is Map ? raw['features'] as Map : const {};
    final ws =
        features['desktop_gateway_ws'] == true ? 'WS ready' : 'WS unknown';
    return 'Hermes $version · $authRequired · $ws';
  }
}

class _StatusText extends StatelessWidget {
  const _StatusText({
    required this.icon,
    required this.tone,
    required this.title,
    required this.body,
  });

  final IconData icon;
  final HermesBadgeTone tone;
  final String title;
  final String body;

  @override
  Widget build(BuildContext context) {
    final colors = context.hermesColors;
    final toneColor = switch (tone) {
      HermesBadgeTone.neutral => colors.accent,
      HermesBadgeTone.success => colors.success,
      HermesBadgeTone.warning => colors.warning,
      HermesBadgeTone.danger => colors.danger,
      HermesBadgeTone.warm => colors.warm,
    };
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Icon(icon, color: toneColor, size: 18),
        const SizedBox(width: 12),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(title, style: Theme.of(context).textTheme.titleSmall),
              const SizedBox(height: 4),
              Text(body, style: TextStyle(color: colors.secondaryText)),
            ],
          ),
        ),
      ],
    );
  }
}

class _ArchitectureNote extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final colors = context.hermesColors;
    return HermesPanel(
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          HermesGlyph(size: 26),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('Local-first by design',
                    style: Theme.of(context).textTheme.titleSmall),
                const SizedBox(height: 5),
                Text(
                  'This client talks directly to your Hermes dashboard over LAN, Tailscale, or a tunnel. Sessions, tools, memory, and secrets stay on the host.',
                  style: TextStyle(color: colors.secondaryText),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
