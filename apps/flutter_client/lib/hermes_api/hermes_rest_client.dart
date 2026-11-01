import 'package:dio/dio.dart';

import 'connection_profile.dart';
import 'url_normalizer.dart';

class HermesStatus {
  const HermesStatus({
    required this.raw,
    this.version,
    this.gatewayRunning,
    this.gatewayState,
    this.activeSessions,
    this.authRequired,
    this.authProviders = const [],
  });

  final Map<String, Object?> raw;
  final String? version;
  final bool? gatewayRunning;
  final String? gatewayState;
  final int? activeSessions;
  final bool? authRequired;
  final List<String> authProviders;

  factory HermesStatus.fromJson(Map<String, Object?> json) {
    return HermesStatus(
      raw: json,
      version: json['version'] as String?,
      gatewayRunning: json['gateway_running'] as bool?,
      gatewayState: json['gateway_state'] as String?,
      activeSessions: json['active_sessions'] as int?,
      authRequired: json['auth_required'] as bool?,
      authProviders: switch (json['auth_providers']) {
        final List<Object?> values => values.whereType<String>().toList(),
        _ => const [],
      },
    );
  }
}

class HermesMobileBootstrap {
  const HermesMobileBootstrap({required this.raw});

  final Map<String, Object?> raw;

  factory HermesMobileBootstrap.fromJson(Map<String, Object?> json) {
    return HermesMobileBootstrap(raw: json);
  }
}

class HermesRestClient {
  HermesRestClient(this.profile, {Dio? dio})
      : _dio = dio ??
            Dio(
              BaseOptions(
                connectTimeout: const Duration(seconds: 8),
                receiveTimeout: const Duration(seconds: 20),
              ),
            );

  final ConnectionProfile profile;
  final Dio _dio;

  Future<HermesStatus> getStatus() async {
    final response = await _dio.getUri<Map<String, Object?>>(
      buildApiUri(profile.baseUrl, '/api/status'),
      options: _options(includeToken: false),
    );
    return HermesStatus.fromJson(response.data ?? const {});
  }

  Future<HermesMobileBootstrap> getMobileBootstrap() async {
    final response = await _dio.getUri<Map<String, Object?>>(
      buildApiUri(profile.baseUrl, '/api/mobile/bootstrap'),
      options: _options(),
    );
    return HermesMobileBootstrap.fromJson(response.data ?? const {});
  }

  Options _options({bool includeToken = true}) {
    final token = profile.token;
    return Options(
      headers: {
        if (includeToken && token != null && token.isNotEmpty)
          'Authorization': 'Bearer $token',
      },
    );
  }
}
