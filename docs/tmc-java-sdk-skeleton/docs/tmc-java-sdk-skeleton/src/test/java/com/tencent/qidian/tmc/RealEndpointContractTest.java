package com.tencent.qidian.tmc;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.tencent.qidian.tmc.modules.analytics.DefaultAnalyticsModule;
import com.tencent.qidian.tmc.modules.customer.DefaultCustomerModule;
import com.tencent.qidian.tmc.modules.segment.DefaultSegmentModule;
import com.tencent.qidian.tmc.modules.tag.DefaultTagModule;
import com.tencent.qidian.tmc.webhook.DefaultWebhookEventNormalizer;
import com.tencent.qidian.tmc.webhook.RawWebhookPayload;
import com.tencent.qidian.tmc.webhook.TmcWebhookEvent;
import com.tencent.qidian.tmc.webhook.TmcWebhookEventType;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.Arrays;
import org.junit.jupiter.api.Test;

/**
 * Contract tests for real endpoint mappings and webhook normalization.
 */
public class RealEndpointContractTest {

    @Test
    void shouldExposeRealOpenApiPaths() {
        assertEquals("/cdp-entity/user/create", DefaultCustomerModule.CREATE_PATH);
        assertEquals("/cdp-entity/user/queryList", DefaultCustomerModule.QUERY_LIST_PATH);
        assertEquals("/cdp-tag/inner-api/open/cdp/tag/tagDefine/external/create", DefaultTagModule.CREATE_EXTERNAL_TAG_PATH);
        assertEquals("/cdp-crowd/import/v2", DefaultSegmentModule.CREATE_IMPORT_SEGMENT_V2_PATH);
        assertEquals("/apiserver/openapi/panels", DefaultAnalyticsModule.LIST_PANELS_PATH);
    }

    @Test
    void shouldNormalizeRealtimeTagWebhook() throws Exception {
        String json = "{\"eventName\":\"realtime_tag_updated\",\"eventTime\":1717395190123,\"bizId\":\"b1\",\"oneId\":\"o1\",\"tagInfo\":{\"id\":1,\"name\":\"VIP\"}}";
        TmcWebhookEvent event = new DefaultWebhookEventNormalizer().normalize(
                new RawWebhookPayload(new ObjectMapper().readTree(json), json));
        assertEquals(TmcWebhookEventType.REALTIME_TAG_CHANGED.getCode(), event.getEventType());
        assertEquals("o1", event.getSubjectId());
        assertEquals("tag", event.getSource());
        assertEquals(Long.valueOf(1717395190123L), event.getEventTime());
        assertTrue(event.getAttributes().containsKey("tagInfo"));
    }

    @Test
    void shouldNormalizeBatchSegmentWebhookList() throws Exception {
        String json = "[{\"eventType\":\"batch_segment_created\",\"timestamp\":\"2026-03-11T14:30:00+08:00\",\"data\":{\"crowdId\":\"c1\",\"crowdName\":\"分群A\"}}]";
        TmcWebhookEvent event = new DefaultWebhookEventNormalizer().normalizeAll(
                new RawWebhookPayload(new ObjectMapper().readTree(json), json)).get(0);
        assertEquals(TmcWebhookEventType.BATCH_SEGMENT_CREATED.getCode(), event.getEventType());
        assertEquals("c1", event.getSubjectId());
        assertEquals("segment", event.getSource());
    }

    @Test
    void shouldPreserveUnknownDottedEventType() throws Exception {
        String json = "{\"eventType\":\"segment.export.completed\",\"eventId\":\"evt-1\"}";
        TmcWebhookEvent event = new DefaultWebhookEventNormalizer().normalize(
                new RawWebhookPayload(new ObjectMapper().readTree(json), json));
        assertEquals("segment.export.completed", event.getEventType());
    }

    @Test
    void shouldExposeKnownWebhookEventCodes() {
        assertTrue(Arrays.asList(TmcWebhookEventType.values()).contains(TmcWebhookEventType.BEHAVIOR_TRACK));
        assertTrue(Arrays.asList(TmcWebhookEventType.values()).contains(TmcWebhookEventType.MEMBER_COUPON_CHANGED));
    }
}
