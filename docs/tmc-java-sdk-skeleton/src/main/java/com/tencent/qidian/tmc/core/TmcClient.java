package com.tencent.qidian.tmc.core;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.tencent.qidian.tmc.auth.HmacSha256ApiSigner;
import com.tencent.qidian.tmc.modules.analytics.AnalyticsModule;
import com.tencent.qidian.tmc.modules.analytics.DefaultAnalyticsModule;
import com.tencent.qidian.tmc.modules.customer.CustomerModule;
import com.tencent.qidian.tmc.modules.customer.DefaultCustomerModule;
import com.tencent.qidian.tmc.modules.segment.DefaultSegmentModule;
import com.tencent.qidian.tmc.modules.segment.SegmentModule;
import com.tencent.qidian.tmc.modules.tag.DefaultTagModule;
import com.tencent.qidian.tmc.modules.tag.TagModule;
import java.util.concurrent.TimeUnit;
import okhttp3.OkHttpClient;

/**
 * High-level SDK client.
 */
public class TmcClient {
    private final CustomerModule customer;
    private final TagModule tag;
    private final SegmentModule segment;
    private final AnalyticsModule analytics;

    public TmcClient(CustomerModule customer,
                     TagModule tag,
                     SegmentModule segment,
                     AnalyticsModule analytics) {
        this.customer = customer;
        this.tag = tag;
        this.segment = segment;
        this.analytics = analytics;
    }

    public static TmcClient create(TmcClientConfig config) {
        OkHttpClient okHttpClient = new OkHttpClient.Builder()
                .connectTimeout(config.getConnectTimeoutMillis(), TimeUnit.MILLISECONDS)
                .readTimeout(config.getReadTimeoutMillis(), TimeUnit.MILLISECONDS)
                .build();
        ObjectMapper objectMapper = new ObjectMapper();
        TmcHttpClient httpClient = new OkHttpTmcHttpClient(config, okHttpClient, objectMapper,
                new HmacSha256ApiSigner());
        return new TmcClient(
                new DefaultCustomerModule(httpClient),
                new DefaultTagModule(httpClient),
                new DefaultSegmentModule(httpClient),
                new DefaultAnalyticsModule(httpClient)
        );
    }

    public CustomerModule customer() {
        return customer;
    }

    public TagModule tag() {
        return tag;
    }

    public SegmentModule segment() {
        return segment;
    }

    public AnalyticsModule analytics() {
        return analytics;
    }
}
