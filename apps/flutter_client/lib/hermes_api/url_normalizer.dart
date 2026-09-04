Uri normalizeHermesBaseUri(String input) {
  final trimmed = input.trim();
  if (trimmed.isEmpty) {
    throw const FormatException('Hermes URL is required.');
  }

  final withScheme = trimmed.contains('://') ? trimmed : 'http://$trimmed';
  final parsed = Uri.parse(withScheme);
  if (!parsed.hasScheme || parsed.host.isEmpty) {
    throw FormatException('Invalid Hermes URL: $input');
  }
  if (parsed.scheme != 'http' && parsed.scheme != 'https') {
    throw FormatException('Hermes URL must use http or https: $input');
  }

  return Uri(
    scheme: parsed.scheme,
    userInfo: parsed.userInfo,
    host: parsed.host,
    port: parsed.hasPort ? parsed.port : null,
    path: _stripTrailingSlash(parsed.path),
  );
}

Uri buildApiUri(Uri baseUri, String path, [Map<String, String>? query]) {
  final basePath = _stripTrailingSlash(baseUri.path);
  final apiPath = path.startsWith('/') ? path : '/$path';
  return Uri(
    scheme: baseUri.scheme,
    userInfo: baseUri.userInfo,
    host: baseUri.host,
    port: baseUri.hasPort ? baseUri.port : null,
    path: '$basePath$apiPath',
    queryParameters: query == null || query.isEmpty ? null : query,
  );
}

Uri buildHermesWsUri(Uri baseUri, {String? token}) {
  final scheme = baseUri.scheme == 'https' ? 'wss' : 'ws';
  final wsBase = Uri(
    scheme: scheme,
    userInfo: baseUri.userInfo,
    host: baseUri.host,
    port: baseUri.hasPort ? baseUri.port : null,
    path: baseUri.path,
  );
  return buildApiUri(wsBase, '/api/ws', {
    if (token != null && token.isNotEmpty) 'token': token,
  });
}

String _stripTrailingSlash(String value) {
  if (value == '/' || value.isEmpty) {
    return '';
  }
  return value.endsWith('/') ? value.substring(0, value.length - 1) : value;
}
