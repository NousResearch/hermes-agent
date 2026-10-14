"""Select an adapter's block transport without changing shared stream semantics."""

from gateway.stream_consumer import GatewayStreamConsumer


def create_stream_consumer(**kwargs):
    adapter = kwargs["adapter"]
    if getattr(adapter, "SUPPORTS_BLOCK_STREAMING", False) is True:
        from gateway.platforms.weixin_streaming import WeixinBlockConsumer
        return WeixinBlockConsumer(**kwargs)
    return GatewayStreamConsumer(**kwargs)
