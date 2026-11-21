import 'dart:async';
import 'dart:convert';

import 'package:web_socket_channel/web_socket_channel.dart';

import 'connection_profile.dart';
import 'url_normalizer.dart';

class HermesGatewayEvent {
  const HermesGatewayEvent({required this.type, this.payload});

  final String type;
  final Map<String, Object?>? payload;
}

class HermesRpcClient {
  HermesRpcClient(this.profile, {WebSocketChannel Function(Uri uri)? connector})
      : _connector = connector ?? WebSocketChannel.connect;

  final ConnectionProfile profile;
  final WebSocketChannel Function(Uri uri) _connector;
  final _events = StreamController<HermesGatewayEvent>.broadcast();

  WebSocketChannel? _channel;
  int _nextId = 1;
  Map<String, Object?>? gatewayReadyPayload;

  Stream<HermesGatewayEvent> get events => _events.stream;
  bool get isConnected => _channel != null;

  Future<void> connect() async {
    if (_channel != null) {
      return;
    }
    final uri = buildHermesWsUri(profile.baseUrl, token: profile.token);
    final channel = _connector(uri);
    _channel = channel;
    channel.stream.listen(
      _handleFrame,
      onError: _events.addError,
      onDone: () => _channel = null,
      cancelOnError: false,
    );
  }

  void sendRequest(String method, [Map<String, Object?> params = const {}]) {
    final channel = _channel;
    if (channel == null) {
      throw StateError('Hermes RPC socket is not connected.');
    }
    channel.sink.add(
      jsonEncode({
        'jsonrpc': '2.0',
        'id': _nextId++,
        'method': method,
        'params': params,
      }),
    );
  }

  Future<void> close() async {
    final channel = _channel;
    _channel = null;
    await channel?.sink.close();
    await _events.close();
  }

  void _handleFrame(Object? frame) {
    if (frame is! String || frame.trim().isEmpty) {
      return;
    }
    final decoded = jsonDecode(frame);
    if (decoded is! Map<String, Object?>) {
      return;
    }
    if (decoded['method'] != 'event') {
      return;
    }
    final params = decoded['params'];
    if (params is! Map<String, Object?>) {
      return;
    }
    final type = params['type'];
    if (type is! String) {
      return;
    }
    final payload = params['payload'] is Map<String, Object?>
        ? params['payload'] as Map<String, Object?>
        : null;
    if (type == 'gateway.ready') {
      gatewayReadyPayload = payload ?? const {};
    }
    _events.add(HermesGatewayEvent(type: type, payload: payload));
  }
}
