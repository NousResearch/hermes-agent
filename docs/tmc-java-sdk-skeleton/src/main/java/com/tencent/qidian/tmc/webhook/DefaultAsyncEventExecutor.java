package com.tencent.qidian.tmc.webhook;

import com.tencent.qidian.tmc.exception.TmcAsyncProcessingException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Default async executor backed by a fixed thread pool.
 */
public class DefaultAsyncEventExecutor implements AsyncEventExecutor {
    private final ExecutorService executorService;

    public DefaultAsyncEventExecutor(int threads) {
        this.executorService = Executors.newFixedThreadPool(threads);
    }

    @Override
    public void submit(TmcWebhookEvent event, final Runnable task) {
        executorService.submit(new Runnable() {
            @Override
            public void run() {
                try {
                    task.run();
                } catch (Exception e) {
                    throw new TmcAsyncProcessingException("Webhook async execution failed.", e);
                }
            }
        });
    }
}
