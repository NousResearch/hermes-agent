package com.tencent.qidian.tmc;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.tencent.qidian.tmc.auth.HmacSha256WebhookSigner;
import com.tencent.qidian.tmc.webhook.AsyncEventExecutor;
import com.tencent.qidian.tmc.webhook.DefaultWebhookDispatcher;
import com.tencent.qidian.tmc.webhook.DefaultWebhookEventNormalizer;
import com.tencent.qidian.tmc.webhook.DefaultWebhookVerifier;
import com.tencent.qidian.tmc.webhook.JacksonWebhookParser;
import com.tencent.qidian.tmc.webhook.TmcWebhookEvent;
import com.tencent.qidian.tmc.webhook.UnknownWebhookEventHandler;
import com.tencent.qidian.tmc.webhook.WebhookEndpoint;
import com.tencent.qidian.tmc.webhook.WebhookEventHandler;
import com.tencent.qidian.tmc.webhook.WebhookHandlerRegistry;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;

/**
 * Webhook endpoint integration test.
 */
public class WebhookEndpointTest {
    @Test
    void shouldVerifyParseNormalizeAndDispatch() {
        HmacSha256WebhookSigner signer = new HmacSha256WebhookSigner();
        String url = "/webhook/tmc";
        String curTime = "1711111111111";
        String sign = signer.signCallback(url, curTime, "secret");

        Map<String, String> query = new HashMap<String, String>();
        query.put("cur_time", curTime);
        query.put("sign", sign);

        final AtomicReference<String> handledType = new AtomicReference<String>();
        WebhookHandlerRegistry registry = new WebhookHandlerRegistry();
        registry.register(new WebhookEventHandler<TmcWebhookEvent>() {
            @Override
            public String supportEventType() {
                return "segment.export.completed";
            }

            @Override
            public void handle(TmcWebhookEvent event) {
                handledType.set(event.getEventType());
            }
        });

        WebhookEndpoint endpoint = WebhookEndpoint.builder()
                .verifier(new DefaultWebhookVerifier("secret", signer))
                .parser(new JacksonWebhookParser(new ObjectMapper()))
                .normalizer(new DefaultWebhookEventNormalizer())
                .dispatcher(new DefaultWebhookDispatcher(registry, new UnknownWebhookEventHandler() {
                    @Override
                    public void handle(TmcWebhookEvent event) {
                        handledType.set("unknown");
                    }
                }))
                .asyncExecutor(new AsyncEventExecutor() {
                    @Override
                    public void submit(TmcWebhookEvent event, Runnable task) {
                        task.run();
                    }
                })
                .build();

        endpoint.handle(url, query, "{\"eventType\":\"segment.batch.created\",\"eventId\":\"evt-1\"}");
        assertEquals("segment.batch.created", handledType.get());
    }
}
