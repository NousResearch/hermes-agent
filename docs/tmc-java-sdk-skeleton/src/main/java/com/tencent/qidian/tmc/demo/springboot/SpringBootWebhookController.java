package com.tencent.qidian.tmc.demo.springboot;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.tencent.qidian.tmc.auth.HmacSha256WebhookSigner;
import com.tencent.qidian.tmc.webhook.DefaultAsyncEventExecutor;
import com.tencent.qidian.tmc.webhook.DefaultWebhookDispatcher;
import com.tencent.qidian.tmc.webhook.DefaultWebhookEventNormalizer;
import com.tencent.qidian.tmc.webhook.DefaultWebhookVerifier;
import com.tencent.qidian.tmc.webhook.JacksonWebhookParser;
import com.tencent.qidian.tmc.webhook.TmcWebhookEvent;
import com.tencent.qidian.tmc.webhook.UnknownWebhookEventHandler;
import com.tencent.qidian.tmc.webhook.WebhookEndpoint;
import com.tencent.qidian.tmc.webhook.WebhookEventHandler;
import com.tencent.qidian.tmc.webhook.WebhookHandlerRegistry;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

/**
 * Example Spring Boot webhook controller.
 */
@RestController
@RequestMapping("/webhook/tmc")
public class SpringBootWebhookController {
    private static final Logger log = LoggerFactory.getLogger(SpringBootWebhookController.class);

    private final WebhookEndpoint endpoint;

    public SpringBootWebhookController() {
        WebhookHandlerRegistry registry = new WebhookHandlerRegistry();
        registry.register(new WebhookEventHandler<TmcWebhookEvent>() {
            @Override
            public String supportEventType() {
                return "unknown";
            }

            @Override
            public void handle(TmcWebhookEvent event) {
                log.info("Received normalized event type={}, id={}", event.getEventType(), event.getEventId());
            }
        });
        UnknownWebhookEventHandler fallback = new UnknownWebhookEventHandler() {
            @Override
            public void handle(TmcWebhookEvent event) {
                log.warn("Unhandled event type={}, payload={}", event.getEventType(), event.getRawPayload());
            }
        };
        this.endpoint = WebhookEndpoint.builder()
                .verifier(new DefaultWebhookVerifier("replace-with-secret", new HmacSha256WebhookSigner()))
                .parser(new JacksonWebhookParser(new ObjectMapper()))
                .normalizer(new DefaultWebhookEventNormalizer())
                .dispatcher(new DefaultWebhookDispatcher(registry, fallback))
                .asyncExecutor(new DefaultAsyncEventExecutor(4))
                .build();
    }

    @PostMapping
    public ResponseEntity<String> receive(@RequestParam Map<String, String> queryParams,
                                          @RequestBody String rawBody) {
        endpoint.handle("/webhook/tmc", queryParams, rawBody);
        return ResponseEntity.ok("success");
    }
}
