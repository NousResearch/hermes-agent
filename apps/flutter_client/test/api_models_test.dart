import 'package:flutter_test/flutter_test.dart';
import 'package:hermes_flutter_client/hermes_api/connection_profile.dart';
import 'package:hermes_flutter_client/hermes_api/hermes_rest_client.dart';

void main() {
  test('connection profile normalizes form input', () {
    final profile = ConnectionProfile.fromForm(
      name: '  Home Hermes  ',
      baseUrl: 'https://host.test/hermes/',
      token: ' secret ',
      now: DateTime.utc(2026, 1, 2),
    );

    expect(profile.name, 'Home Hermes');
    expect(profile.baseUrl.toString(), 'https://host.test/hermes');
    expect(profile.token, 'secret');
    expect(profile.createdAt, DateTime.utc(2026, 1, 2));
  });

  test('status tolerates partial dashboard payloads', () {
    final status = HermesStatus.fromJson({
      'gateway_running': true,
      'gateway_state': 'running',
      'active_sessions': 3,
      'auth_required': true,
      'auth_providers': ['basic', 'oauth'],
    });

    expect(status.gatewayRunning, isTrue);
    expect(status.gatewayState, 'running');
    expect(status.activeSessions, 3);
    expect(status.authRequired, isTrue);
    expect(status.authProviders, ['basic', 'oauth']);
  });
}
