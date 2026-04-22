package com.tencent.qidian.tmc.webhook;

/**
 * Default dispatcher using a handler registry and fallback handler.
 */
public class DefaultWebhookDispatcher implements WebhookDispatcher {
    private final WebhookHandlerRegistry registry;
    private final UnknownWebhookEventHandler unknownHandler;

    public DefaultWebhookDispatcher(WebhookHandlerRegistry registry,
                                    UnknownWebhookEventHandler unknownHandler) {
        this.registry = registry;
        this.unknownHandler = unknownHandler;
    }

    @Override
    @SuppressWarnings("unchecked")
    public void dispatch(TmcWebhookEvent event) {
        WebhookEventHandler<?> handler = registry.find(event.getEventType()).orElse(null);
        if (handler == null) {
            unknownHandler.handle(event);
            return;
        }
        ((WebhookEventHandler<TmcWebhookEvent>) handler).handle(event);
    }
}
