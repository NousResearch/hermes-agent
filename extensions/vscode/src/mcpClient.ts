/**
 * Hermes MCP Client — communicates with `hermes mcp serve` over stdio JSON-RPC.
 *
 * Handles:
 * - Process spawn and lifecycle
 * - JSON-RPC 2.0 request/response
 * - Automatic reconnection with exponential backoff
 * - Tool discovery (conversation list, message send)
 */

import { ChildProcess, spawn } from 'child_process';
import { createInterface, Interface } from 'readline';
import * as vscode from 'vscode';

interface MCPClientConfig {
    command: string;
    args: string[];
    outputChannel: vscode.OutputChannel;
    autoConnect: boolean;
}

interface JSONRPCRequest {
    jsonrpc: '2.0';
    id: number;
    method: string;
    params: Record<string, unknown>;
}

interface JSONRPCResponse {
    jsonrpc: '2.0';
    id: number;
    result?: unknown;
    error?: { code: number; message: string; data?: unknown };
}

type PendingCall = {
    resolve: (value: string) => void;
    reject: (error: Error) => void;
    timer: NodeJS.Timeout;
};

export class HermesMCPClient {
    private config: MCPClientConfig;
    private process: ChildProcess | null = null;
    private rl: Interface | null = null;
    private nextId = 1;
    private pending = new Map<number, PendingCall>();
    private _connected = false;
    private reconnectAttempts = 0;
    private reconnectTimer: NodeJS.Timeout | null = null;
    private readonly MAX_RECONNECT_DELAY = 30_000;

    constructor(config: MCPClientConfig) {
        this.config = config;
    }

    isConnected(): boolean {
        return this._connected && this.process !== null && !this.process.killed;
    }

    async connect(): Promise<void> {
        if (this.isConnected()) { return; }

        this.config.outputChannel.appendLine(
            `Starting Hermes MCP: ${this.config.command} ${this.config.args.join(' ')}`
        );

        return new Promise((resolve, reject) => {
            this.process = spawn(this.config.command, this.config.args, {
                stdio: ['pipe', 'pipe', 'pipe'],
                env: { ...process.env },
            });

            let initialized = false;

            this.process.on('error', (err) => {
                this.config.outputChannel.appendLine(`Process error: ${err.message}`);
                if (!initialized) { reject(err); }
            });

            this.process.on('exit', (code, signal) => {
                this.config.outputChannel.appendLine(
                    `Hermes process exited (code=${code}, signal=${signal})`
                );
                this._connected = false;
                if (!initialized) {
                    reject(new Error(`Process exited with code ${code}`));
                }
                this.scheduleReconnect();
            });

            if (this.process.stderr) {
                // Hermes logs to stderr in MCP serve mode
                const stderrRl = createInterface({ input: this.process.stderr });
                stderrRl.on('line', (line) => {
                    this.config.outputChannel.appendLine(`[hermes] ${line}`);
                });
            }

            if (this.process.stdout) {
                this.rl = createInterface({ input: this.process.stdout });
                this.rl.on('line', (line) => {
                    try {
                        const msg = JSON.parse(line) as JSONRPCResponse;
                        if (msg.id !== undefined && this.pending.has(msg.id)) {
                            const pending = this.pending.get(msg.id)!;
                            clearTimeout(pending.timer);
                            this.pending.delete(msg.id);
                            if (msg.error) {
                                pending.reject(new Error(msg.error.message));
                            } else {
                                pending.resolve(
                                    typeof msg.result === 'string'
                                        ? msg.result
                                        : JSON.stringify(msg.result)
                                );
                            }
                        }
                    } catch {
                        // Non-JSON line (e.g. initialization messages)
                    }
                });
            }

            // Give the process a moment to start
            setTimeout(() => {
                if (!initialized && this.process && !this.process.killed) {
                    initialized = true;
                    this._connected = true;
                    this.reconnectAttempts = 0;
                    this.config.outputChannel.appendLine('Hermes MCP connected.');
                    resolve();
                }
            }, 1000);
        });
    }

    disconnect(): void {
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
        if (this.process && !this.process.killed) {
            this.process.kill();
        }
        this._connected = false;
        this.config.outputChannel.appendLine('Hermes MCP disconnected.');
    }

    private scheduleReconnect(): void {
        if (!this.config.autoConnect) { return; }

        const delay = Math.min(
            1000 * Math.pow(2, this.reconnectAttempts),
            this.MAX_RECONNECT_DELAY
        );
        this.reconnectAttempts++;

        this.config.outputChannel.appendLine(
            `Reconnecting in ${delay / 1000}s (attempt ${this.reconnectAttempts})...`
        );

        this.reconnectTimer = setTimeout(async () => {
            try {
                await this.connect();
                this.config.outputChannel.appendLine('Reconnected successfully.');
            } catch {
                this.config.outputChannel.appendLine('Reconnect failed, will retry...');
                this.scheduleReconnect();
            }
        }, delay);
    }

    async sendMessage(message: string, sessionKey?: string): Promise<string> {
        if (!this.isConnected()) {
            throw new Error('Not connected to Hermes MCP server');
        }

        // Use the MCP messages_send tool
        const result = await this.callTool(
            sessionKey ? 'messages_send' : 'conversations_list',
            sessionKey
                ? { target: sessionKey, message }
                : { limit: 10 }
        );

        // If we got conversation list, send to the first/recent conversation
        if (!sessionKey) {
            try {
                const list = JSON.parse(result);
                if (list.conversations && list.conversations.length > 0) {
                    const target = list.conversations[0].session_key;
                    return this.callTool('messages_send', { target, message });
                }
            } catch {
                // Fall through
            }
        }

        return result;
    }

    private callTool(method: string, params: Record<string, unknown>): Promise<string> {
        return new Promise((resolve, reject) => {
            const id = this.nextId++;
            const request: JSONRPCRequest = {
                jsonrpc: '2.0',
                id,
                method: `tools/call`,
                params: {
                    name: method,
                    arguments: params,
                },
            };

            const timer = setTimeout(() => {
                this.pending.delete(id);
                reject(new Error(`MCP call timed out: ${method}`));
            }, 60_000); // 60 second timeout

            this.pending.set(id, { resolve, reject, timer });

            if (this.process?.stdin) {
                this.process.stdin.write(JSON.stringify(request) + '\n');
            } else {
                clearTimeout(timer);
                this.pending.delete(id);
                reject(new Error('Process stdin not available'));
            }
        });
    }

    cancelCurrent(): void {
        // Clear all pending calls
        for (const [id, pending] of this.pending) {
            clearTimeout(pending.timer);
            pending.reject(new Error('Cancelled by user'));
        }
        this.pending.clear();
    }
}
