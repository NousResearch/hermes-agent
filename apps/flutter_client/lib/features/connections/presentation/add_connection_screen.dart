import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../../design/hermes_components.dart';
import '../../../design/hermes_theme.dart';
import '../../../hermes_api/connection_profile.dart';
import '../data/connection_store.dart';

class AddConnectionScreen extends ConsumerStatefulWidget {
  const AddConnectionScreen({super.key});

  @override
  ConsumerState<AddConnectionScreen> createState() =>
      _AddConnectionScreenState();
}

class _AddConnectionScreenState extends ConsumerState<AddConnectionScreen> {
  final _formKey = GlobalKey<FormState>();
  final _name = TextEditingController();
  final _url = TextEditingController(text: 'http://127.0.0.1:9119');
  final _token = TextEditingController();
  String? _error;

  @override
  void dispose() {
    _name.dispose();
    _url.dispose();
    _token.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final colors = context.hermesColors;
    return HermesShell(
      title: 'Add Instance',
      subtitle: 'Manual URL + token today; QR pairing comes next.',
      leading: IconButton(
        tooltip: 'Back',
        onPressed: () => context.go('/'),
        icon: const Icon(Icons.arrow_back),
      ),
      children: [
        HermesPanel(
          child: Form(
            key: _formKey,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Text(
                  'Connection profile',
                  style: Theme.of(context).textTheme.titleMedium,
                ),
                const SizedBox(height: 12),
                TextFormField(
                  controller: _name,
                  decoration: const InputDecoration(
                    labelText: 'Name',
                    hintText: 'Laptop Hermes',
                    prefixIcon: Icon(Icons.badge_outlined),
                  ),
                ),
                const SizedBox(height: 12),
                TextFormField(
                  controller: _url,
                  decoration: const InputDecoration(
                    labelText: 'Dashboard URL',
                    hintText: 'https://hermes.tailnet-name.ts.net',
                    prefixIcon: Icon(Icons.link),
                  ),
                  keyboardType: TextInputType.url,
                  validator: (value) => value == null || value.trim().isEmpty
                      ? 'URL required'
                      : null,
                ),
                const SizedBox(height: 12),
                TextFormField(
                  controller: _token,
                  decoration: const InputDecoration(
                    labelText: 'Session token',
                    hintText:
                        'Optional for loopback; required for token-mode hosts',
                    prefixIcon: Icon(Icons.key_outlined),
                  ),
                  obscureText: true,
                ),
                if (_error != null) ...[
                  const SizedBox(height: 12),
                  HermesPanel(
                    padding: const EdgeInsets.all(10),
                    child: Row(
                      children: [
                        Icon(Icons.error_outline,
                            color: colors.danger, size: 17),
                        const SizedBox(width: 8),
                        Expanded(
                          child: Text(_error!,
                              style: TextStyle(color: colors.danger)),
                        ),
                      ],
                    ),
                  ),
                ],
                const SizedBox(height: 18),
                FilledButton.icon(
                  onPressed: _save,
                  icon: const Icon(Icons.save_outlined, size: 16),
                  label: const Text('Save connection'),
                ),
              ],
            ),
          ),
        ),
        HermesPanel(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const HermesBadge(
                label: 'Recommended path',
                icon: Icons.lock_outline,
                tone: HermesBadgeTone.warm,
              ),
              const SizedBox(height: 10),
              Text('Use Tailscale or Cloudflare Tunnel',
                  style: Theme.of(context).textTheme.titleSmall),
              const SizedBox(height: 5),
              Text(
                'The mobile app should reach your own Hermes dashboard directly. No hosted relay is required for this MVP.',
                style: TextStyle(color: colors.secondaryText),
              ),
            ],
          ),
        ),
      ],
    );
  }

  void _save() {
    if (!_formKey.currentState!.validate()) {
      return;
    }
    try {
      final profile = ConnectionProfile.fromForm(
        name: _name.text,
        baseUrl: _url.text,
        token: _token.text,
      );
      ref.read(connectionStoreProvider.notifier).add(profile);
      ref.read(selectedConnectionIdProvider.notifier).state = profile.id;
      context.go('/');
    } on FormatException catch (error) {
      setState(() => _error = error.message);
    }
  }
}
