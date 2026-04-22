package com.tencent.qidian.tmc.webhook;

import com.fasterxml.jackson.databind.JsonNode;

/**
 * Raw webhook payload after JSON parsing.
 */
public class RawWebhookPayload {
    private final JsonNode body;
    private final String rawBody;

    public RawWebhookPayload(JsonNode body) {
        this(body, body == null ? null : body.toString());
    }

    public RawWebhookPayload(JsonNode body, String rawBody) {
        this.body = body;
        this.rawBody = rawBody;
    }

    public JsonNode getBody() {
        return body;
    }

    public String getRawBody() {
        return rawBody;
    }
}
