package com.tencent.qidian.tmc.webhook;

/**
 * Executes webhook tasks asynchronously.
 */
public interface AsyncEventExecutor {
    void submit(TmcWebhookEvent event, Runnable task);
}
