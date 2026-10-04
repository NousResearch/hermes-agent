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


@pytest.mark.asyncio
async def test_pipeline_persists_partial_record_and_reloads_before_retry(tmp_path):
    from plugins.teams_pipeline.pipeline import TeamsMeetingPipeline
    from plugins.teams_pipeline.store import TeamsPipelineStore
    from plugins.teams_pipeline.models import TeamsMeetingRef, TeamsMeetingSummaryPayload
    calls = []
    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(500 if len(calls) == 2 else 202)
    writer = TeamsSummaryWriter(transport=httpx.MockTransport(handler))
    store = TeamsPipelineStore(tmp_path / 'store.json')
    pipeline = TeamsMeetingPipeline(graph_client=object(), store=store, teams_sender=writer,
        config={'teams_delivery': {'enabled': True, 'mode': 'incoming_webhook', 'incoming_webhook_urls': ['https://one.example/hook', 'https://two.example/hook']}})
    job = pipeline.create_job_from_notification({'id': 'event', 'resourceData': {'meetingId': 'meeting'}})
    payload = TeamsMeetingSummaryPayload(meeting_ref=TeamsMeetingRef(meeting_id='meeting'), title='Sync', summary='Summary')
    with pytest.raises(PartialWebhookDeliveryError):
        await pipeline._write_sinks(job, payload)
    pipeline.store = TeamsPipelineStore(store.path)
    assert pipeline.store.get_sink_record('teams:meeting')['delivered'] is False
    await pipeline._write_sinks(job, payload)
    assert TeamsPipelineStore(store.path).get_sink_record('teams:meeting')['delivered'] is True
    assert calls == ['https://one.example/hook', 'https://two.example/hook', 'https://two.example/hook']
