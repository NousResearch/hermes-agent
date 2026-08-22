import 'package:flutter_test/flutter_test.dart';
import 'package:hermes_flutter_client/hermes_api/url_normalizer.dart';

void main() {
  test('adds http scheme when missing', () {
    expect(
      normalizeHermesBaseUri('127.0.0.1:9119').toString(),
      'http://127.0.0.1:9119',
    );
  });

  test('preserves base path for tunneled deployments', () {
    final base = normalizeHermesBaseUri('https://example.com/hermes/');
    expect(buildApiUri(base, '/api/status').toString(),
        'https://example.com/hermes/api/status');
  });

  test('builds websocket URL with encoded token', () {
    final base = normalizeHermesBaseUri('https://gw.example.com');
    expect(
      buildHermesWsUri(base, token: 'a/b c').toString(),
      'wss://gw.example.com/api/ws?token=a%2Fb+c',
    );
  });

  test('rejects unsupported schemes', () {
    expect(
      () => normalizeHermesBaseUri('ftp://example.com'),
      throwsFormatException,
    );
  });
}
