package com.tencent.qidian.tmc.webhook;

/**
 * Canonical webhook event types normalized across documented TMC callbacks.
 */
public enum TmcWebhookEventType {
    REALTIME_TAG_CHANGED("tag.realtime.changed"),
    BATCH_SEGMENT_CREATED("segment.batch.created"),
    MEMBER_COUPON_CHANGED("member.coupon.changed"),
    MEMBER_BENEFIT_CHANGED("member.benefit.changed"),
    BEHAVIOR_TRACK("behavior.track"),
    UNKNOWN("unknown");

    private final String code;

    TmcWebhookEventType(String code) {
        this.code = code;
    }

    public String getCode() {
        return code;
    }

    public static TmcWebhookEventType fromRawType(String rawType) {
        if (rawType == null || rawType.trim().isEmpty()) {
            return UNKNOWN;
        }
        String normalized = rawType.trim().toLowerCase();
        if (normalized.contains("realtime_tag") || normalized.contains("tag_realtime") || normalized.contains("tag.update")) {
            return REALTIME_TAG_CHANGED;
        }
        if (normalized.contains("batch_segment") || normalized.contains("crowd") || normalized.contains("segment")) {
            return BATCH_SEGMENT_CREATED;
        }
        if (normalized.contains("coupon")) {
            return MEMBER_COUPON_CHANGED;
        }
        if (normalized.contains("benefit") || normalized.contains("rights")) {
            return MEMBER_BENEFIT_CHANGED;
        }
        if (normalized.contains("behavior") || normalized.contains("track")) {
            return BEHAVIOR_TRACK;
        }
        return UNKNOWN;
    }
}
