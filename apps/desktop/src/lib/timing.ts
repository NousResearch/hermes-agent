// Streaming view-flush batching for the desktop renderer.
//
// The chat window re-renders the transcript every time a streaming update is
// flushed into the view. Historically the flush was scheduled on the next
// animation frame (60 fps), so a token stream repainted the whole conversation
// at 60 fps — pinning a renderer core on low-power machines (iGPU laptops,
// fanless MacBooks) and competing with local inference on shared iGPUs.
// See https://github.com/NousResearch/hermes-agent/issues/50107.
//
// These windows mirror the TUI's timing module (ui-tui/src/config/timing.ts).
// 80 ms ≈ 12 fps — visually smooth for reading streamed text while cutting
// the flush (and thus re-render) rate ~5x. Idle heartbeats (nothing actively
// streaming) batch harder still: the view is static, only the status line
// ticks.
export const STREAM_BATCH_MS = 80
export const STREAM_IDLE_BATCH_MS = 200
