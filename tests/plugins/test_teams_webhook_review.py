from types import SimpleNamespace
import httpx
import pytest
from gateway.config import Platform, PlatformConfig
from plugins.teams_pipeline.runtime import build_pipeline_runtime_config
from plugins.platforms.teams.summary_writer import TeamsSummaryWriter, PartialWebhookDeliveryError


def test_yaml_list_runtime():
    urls = ['https://one.example/hook', 'https://two.example/hook']
    config = SimpleNamespace(platforms={Platform('teams'): PlatformConfig(enabled=True, extra={'delivery_mode': 'incoming_webhook', 'incoming_webhook_urls': urls})})
    delivery = build_pipeline_runtime_config(config)['teams_delivery']
    assert delivery['enabled'] and delivery['incoming_webhook_urls'] == urls


@pytest.mark.asyncio
async def test_partial_retry_and_plural_autodetection():
    calls = []
    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(500 if len(calls) == 2 else 202)
    writer = TeamsSummaryWriter(transport=httpx.MockTransport(handler))
    payload = SimpleNamespace(title='Sync', summary='Summary', key_decisions=[], action_items=[], risks=[])
    config = {'incoming_webhook_urls': ['https://one.example/hook', 'https://two.example/hook']}
    with pytest.raises(PartialWebhookDeliveryError) as error:
        await writer.write_summary(payload, config)
    assert not error.value.delivery_record['delivered']
    result = await writer.write_summary(payload, config, error.value.delivery_record)
    assert result['delivered']
    assert calls == ['https://one.example/hook', 'https://two.example/hook', 'https://two.example/hook']
