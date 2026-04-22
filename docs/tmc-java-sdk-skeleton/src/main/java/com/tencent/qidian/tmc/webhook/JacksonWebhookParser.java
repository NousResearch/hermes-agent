package com.tencent.qidian.tmc.webhook;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * Jackson-based webhook parser.
 */
public class JacksonWebhookParser implements WebhookParser {
    private final ObjectMapper objectMapper;

    public JacksonWebhookParser(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper;
    }

    @Override
    public RawWebhookPayload parse(String rawBody) {
        try {
            JsonNode node = objectMapper.readTree(rawBody);
            return new RawWebhookPayload(node, rawBody);
        } catch (Exception e) {
            throw new IllegalArgumentException("Failed to parse webhook payload.", e);
        }
    }
}
