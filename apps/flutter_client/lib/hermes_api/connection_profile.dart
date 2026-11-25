import 'url_normalizer.dart';

class ConnectionProfile {
  const ConnectionProfile({
    required this.id,
    required this.name,
    required this.baseUrl,
    this.token,
    this.createdAt,
    this.lastUsedAt,
  });

  final String id;
  final String name;
  final Uri baseUrl;
  final String? token;
  final DateTime? createdAt;
  final DateTime? lastUsedAt;

  factory ConnectionProfile.fromForm({
    required String name,
    required String baseUrl,
    String? token,
    DateTime? now,
  }) {
    final createdAt = now ?? DateTime.now();
    final uri = normalizeHermesBaseUri(baseUrl);
    return ConnectionProfile(
      id: '${uri.host}:${uri.port == 0 ? uri.scheme : uri.port}',
      name: name.trim().isEmpty ? uri.host : name.trim(),
      baseUrl: uri,
      token: token == null || token.trim().isEmpty ? null : token.trim(),
      createdAt: createdAt,
      lastUsedAt: createdAt,
    );
  }

  factory ConnectionProfile.fromJson(Map<String, Object?> json) {
    return ConnectionProfile(
      id: json['id'] as String,
      name: json['name'] as String,
      baseUrl: normalizeHermesBaseUri(json['baseUrl'] as String),
      token: json['token'] as String?,
      createdAt: _date(json['createdAt']),
      lastUsedAt: _date(json['lastUsedAt']),
    );
  }

  Map<String, Object?> toJson() {
    return {
      'id': id,
      'name': name,
      'baseUrl': baseUrl.toString(),
      if (token != null) 'token': token,
      if (createdAt != null) 'createdAt': createdAt!.toIso8601String(),
      if (lastUsedAt != null) 'lastUsedAt': lastUsedAt!.toIso8601String(),
    };
  }

  ConnectionProfile copyWith({DateTime? lastUsedAt}) {
    return ConnectionProfile(
      id: id,
      name: name,
      baseUrl: baseUrl,
      token: token,
      createdAt: createdAt,
      lastUsedAt: lastUsedAt ?? this.lastUsedAt,
    );
  }

  static DateTime? _date(Object? value) {
    return value is String ? DateTime.tryParse(value) : null;
  }
}
