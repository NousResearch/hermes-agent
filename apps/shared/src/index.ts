export {
  JsonRpcGatewayClient,
  type ConnectionState,
  type GatewayClientOptions,
  type GatewayEvent,
  type GatewayEventName,
  type GatewayRequestId,
  type JsonRpcFrame,
  type WebSocketLike
} from './json-rpc-gateway'
export {
  GatewayReauthRequiredError,
  buildHermesWebSocketUrl,
  isGatewayAuthShapedError,
  isGatewayReauthRequired,
  resolveGatewayWsUrl,
  type GatewayAuthMode,
  type GatewayWsConnection,
  type HermesWebSocketUrlOptions,
  type ResolveGatewayWsUrlDeps,
  type WebSocketAuthParam
} from './websocket-url'
