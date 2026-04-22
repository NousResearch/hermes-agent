package com.tencent.qidian.tmc.webhook;

import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Registry for webhook handlers.
 */
public class WebhookHandlerRegistry {
    private final Map<String, WebhookEventHandler<?>> handlers = new ConcurrentHashMap<String, WebhookEventHandler<?>>();

    public void register(WebhookEventHandler<?> handler) {
        handlers.put(handler.supportEventType(), handler);
    }

    public Optional<WebhookEventHandler<?>> find(String eventType) {
        return Optional.ofNullable(handlers.get(eventType));
    }
}
