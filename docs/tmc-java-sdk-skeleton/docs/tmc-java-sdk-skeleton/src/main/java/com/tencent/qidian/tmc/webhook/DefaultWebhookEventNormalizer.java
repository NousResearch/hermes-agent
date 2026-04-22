package com.tencent.qidian.tmc.webhook;

import com.fasterxml.jackson.databind.JsonNode;
import java.time.Instant;
import java.time.OffsetDateTime;
import java.time.format.DateTimeParseException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Normalizes multiple documented TMC webhook payloads into a unified event model.
 */
public class DefaultWebhookEventNormalizer implements WebhookEventNormalizer {
    @Override
    public TmcWebhookEvent normalize(RawWebhookPayload payload) {
        List<TmcWebhookEvent> events = normalizeAll(payload);
        return events.isEmpty() ? createEvent(null) : events.get(0);
    }

    @Override
    public List<TmcWebhookEvent> normalizeAll(RawWebhookPayload payload) {
        List<TmcWebhookEvent> events = new ArrayList<TmcWebhookEvent>();
        JsonNode body = payload.getBody();
        if (body == null || body.isNull()) {
            events.add(createEvent(null));
            return events;
        }
        if (body.isArray()) {
            for (JsonNode item : body) {
                events.add(createEvent(item));
            }
            return events;
        }
        events.add(createEvent(body));
        return events;
    }

    private TmcWebhookEvent createEvent(JsonNode body) {
        TmcWebhookEvent event = new TmcWebhookEvent();
        event.setRawPayload(body);
        event.setHeaders(new LinkedHashMap<String, Object>());
        event.setAttributes(flatten(body));
        event.setEventId(text(body, "eventId", "id", "msgId", "bizId"));
        event.setEventType(resolveEventType(body));
        event.setSource(resolveSource(body, event.getEventType()));
        event.setSubjectId(resolveSubjectId(body));
        event.setEventTime(resolveEventTime(body));
        return event;
    }

    private String resolveEventType(JsonNode body) {
        String raw = text(body, "eventName", "eventType", "type", "msgType");
        TmcWebhookEventType mapped = TmcWebhookEventType.fromRawType(raw);
        if (mapped == TmcWebhookEventType.UNKNOWN && raw != null && raw.contains(".")) {
            return raw;
        }
        return mapped.getCode();
    }

    private String resolveSource(JsonNode body, String eventType) {
        if (eventType == null) {
            return "unknown";
        }
        if (eventType.startsWith("tag.")) {
            return "tag";
        }
        if (eventType.startsWith("segment.")) {
            return "segment";
        }
        if (eventType.startsWith("member.coupon") || eventType.startsWith("member.benefit")) {
            return "member";
        }
        String explicit = text(body, "source", "eventSource", "bizType");
        return explicit == null || explicit.isEmpty() ? "unknown" : explicit;
    }

    private String resolveSubjectId(JsonNode body) {
        String direct = text(body, "subjectId", "oneId", "unionId", "openId", "crowdId", "memberId", "userId");
        if (direct != null) {
            return direct;
        }
        JsonNode data = child(body, "data");
        String nested = text(data, "subjectId", "crowdId", "oneId", "memberId", "userId");
        if (nested != null) {
            return nested;
        }
        JsonNode coupon = child(body, "couponInfo");
        return text(coupon, "memberId", "couponId");
    }

    private Long resolveEventTime(JsonNode body) {
        String[] fields = new String[] {"eventTime", "timestamp", "time", "occurTime", "pushTime"};
        for (String field : fields) {
            JsonNode node = body == null ? null : body.get(field);
            if (node == null || node.isNull()) {
                continue;
            }
            if (node.isNumber()) {
                return node.asLong();
            }
            if (node.isTextual()) {
                String text = node.asText();
                try {
                    return Long.valueOf(text);
                } catch (NumberFormatException ignored) {
                    try {
                        return OffsetDateTime.parse(text).toInstant().toEpochMilli();
                    } catch (DateTimeParseException ignored2) {
                        try {
                            return Instant.parse(text).toEpochMilli();
                        } catch (DateTimeParseException ignored3) {
                            return null;
                        }
                    }
                }
            }
        }
        return null;
    }

    private Map<String, Object> flatten(JsonNode body) {
        Map<String, Object> attributes = new LinkedHashMap<String, Object>();
        if (body == null || body.isNull()) {
            return attributes;
        }
        Iterator<String> fieldNames = body.fieldNames();
        while (fieldNames.hasNext()) {
            String field = fieldNames.next();
            JsonNode value = body.get(field);
            if (value == null || value.isNull()) {
                continue;
            }
            if (value.isValueNode()) {
                attributes.put(field, value.asText());
            } else {
                attributes.put(field, value);
            }
        }
        return attributes;
    }

    private JsonNode child(JsonNode body, String fieldName) {
        if (body == null) {
            return null;
        }
        JsonNode node = body.get(fieldName);
        return node == null || node.isNull() ? null : node;
    }

    private String text(JsonNode body, String... candidates) {
        if (body == null) {
            return null;
        }
        for (String candidate : candidates) {
            JsonNode node = body.get(candidate);
            if (node != null && !node.isNull()) {
                return node.asText();
            }
        }
        return null;
    }
}
