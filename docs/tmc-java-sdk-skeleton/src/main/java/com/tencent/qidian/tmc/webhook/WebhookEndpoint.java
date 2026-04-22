package com.tencent.qidian.tmc.webhook;

import java.util.List;

/**
 * Unified webhook endpoint entry.
 */
public class WebhookEndpoint {
    private final WebhookVerifier verifier;
    private final WebhookParser parser;
    private final WebhookEventNormalizer normalizer;
    private final WebhookDispatcher dispatcher;
    private final AsyncEventExecutor asyncEventExecutor;

    private WebhookEndpoint(Builder builder) {
        this.verifier = builder.verifier;
        this.parser = builder.parser;
        this.normalizer = builder.normalizer;
        this.dispatcher = builder.dispatcher;
        this.asyncEventExecutor = builder.asyncEventExecutor;
    }

    public static Builder builder() {
        return new Builder();
    }

    public void handle(String requestUrl, java.util.Map<String, String> queryParams, String rawBody) {
        WebhookRequest request = new WebhookRequest(requestUrl, queryParams, rawBody);
        verifier.verify(request);
        RawWebhookPayload payload = parser.parse(rawBody);
        List<TmcWebhookEvent> events = normalizer.normalizeAll(payload);
        for (final TmcWebhookEvent event : events) {
            asyncEventExecutor.submit(event, new Runnable() {
                @Override
                public void run() {
                    dispatcher.dispatch(event);
                }
            });
        }
    }

    public static final class Builder {
        private WebhookVerifier verifier;
        private WebhookParser parser;
        private WebhookEventNormalizer normalizer;
        private WebhookDispatcher dispatcher;
        private AsyncEventExecutor asyncEventExecutor;

        public Builder verifier(WebhookVerifier verifier) {
            this.verifier = verifier;
            return this;
        }

        public Builder parser(WebhookParser parser) {
            this.parser = parser;
            return this;
        }

        public Builder normalizer(WebhookEventNormalizer normalizer) {
            this.normalizer = normalizer;
            return this;
        }

        public Builder dispatcher(WebhookDispatcher dispatcher) {
            this.dispatcher = dispatcher;
            return this;
        }

        public Builder asyncExecutor(AsyncEventExecutor asyncEventExecutor) {
            this.asyncEventExecutor = asyncEventExecutor;
            return this;
        }

        public WebhookEndpoint build() {
            return new WebhookEndpoint(this);
        }
    }
}
