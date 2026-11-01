/**
 * Hermes Chat Panel — WebView-based sidebar chat UI.
 *
 * Provides:
 * - Message history with user/assistant/error styling
 * - Typing indicator
 * - Status bar integration
 * - Clear/reset functionality
 */

import * as vscode from 'vscode';

interface ChatMessage {
    role: 'user' | 'assistant' | 'error' | 'system';
    content: string;
    timestamp: number;
    id: string;
}

type ChatStatus = 'idle' | 'thinking' | 'error' | 'disconnected';

export class ChatPanel implements vscode.WebviewViewProvider {
    public static readonly viewType = 'hermes.chatView';
    private _view?: vscode.WebviewView;
    private _extensionUri: vscode.Uri;
    private _outputChannel: vscode.OutputChannel;
    private _messages: ChatMessage[] = [];
    private _status: ChatStatus = 'disconnected';
    private _statusBarItem: vscode.StatusBarItem;
    private _messageIdCounter = 0;

    constructor(extensionUri: vscode.Uri, outputChannel: vscode.OutputChannel) {
        this._extensionUri = extensionUri;
        this._outputChannel = outputChannel;

        this._statusBarItem = vscode.window.createStatusBarItem(
            vscode.StatusBarAlignment.Right,
            100
        );
        this._statusBarItem.command = 'hermes.showChat';
        this.updateStatusBar();

        vscode.window.registerWebviewViewProvider(ChatPanel.viewType, this, {
            webviewOptions: { retainContextWhenHidden: true },
        });
    }

    public resolveWebviewView(
        webviewView: vscode.WebviewView,
    ): void {
        this._view = webviewView;

        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this._extensionUri],
        };

        webviewView.webview.html = this._getHtml();

        webviewView.webview.onDidReceiveMessage((data) => {
            switch (data.type) {
                case 'sendMessage':
                    vscode.commands.executeCommand('hermes.customTask');
                    break;
                case 'ready':
                    this._outputChannel.appendLine('Chat panel webview ready.');
                    break;
            }
        });

        webviewView.onDidChangeVisibility(() => {
            if (webviewView.visible) {
                this.updateStatusBar();
            }
        });
    }

    reveal(): void {
        if (this._view) {
            this._view.show(true);
        }
    }

    addUserMessage(content: string): void {
        this._addMessage({ role: 'user', content });
    }

    addAssistantMessage(content: string): void {
        this._addMessage({ role: 'assistant', content });
    }

    addErrorMessage(content: string): void {
        this._addMessage({ role: 'error', content });
    }

    setStatus(status: ChatStatus): void {
        this._status = status;
        this.updateStatusBar();
        this._postToWebview({ type: 'setStatus', status });
    }

    clear(): void {
        this._messages = [];
        this._postToWebview({ type: 'clear' });
    }

    private _addMessage(msg: Omit<ChatMessage, 'timestamp' | 'id'>): void {
        const full: ChatMessage = {
            ...msg,
            id: `msg-${++this._messageIdCounter}`,
            timestamp: Date.now(),
        };

        // Enforce max history
        const maxHistory = vscode.workspace.getConfiguration('hermes').get<number>('maxHistory', 50);
        while (this._messages.length >= maxHistory) {
            this._messages.shift();
        }
        this._messages.push(full);

        this._postToWebview({
            type: 'addMessage',
            message: full,
        });
    }

    private _postToWebview(message: unknown): void {
        this._view?.webview?.postMessage(message);
    }

    private updateStatusBar(): void {
        const icons: Record<ChatStatus, string> = {
            idle: '$(pass)',
            thinking: '$(sync~spin)',
            error: '$(error)',
            disconnected: '$(debug-disconnect)',
        };

        const labels: Record<ChatStatus, string> = {
            idle: 'Hermes: Ready',
            thinking: 'Hermes: Thinking...',
            error: 'Hermes: Error',
            disconnected: 'Hermes: Disconnected',
        };

        this._statusBarItem.text = `${icons[this._status]} ${labels[this._status]}`;
        this._statusBarItem.tooltip = `Hermes Agent — ${labels[this._status]}`;

        if (this._status === 'error') {
            this._statusBarItem.backgroundColor = new vscode.ThemeColor(
                'statusBarItem.errorBackground'
            );
        } else {
            this._statusBarItem.backgroundColor = undefined;
        }

        this._statusBarItem.show();
    }

    private _getHtml(): string {
        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="Content-Security-Policy"
          content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline';">
    <title>Hermes Chat</title>
    <style>
        :root {
            --bg: var(--vscode-sideBar-background, #1e1e2e);
            --fg: var(--vscode-sideBar-foreground, #cdd6f4);
            --border: var(--vscode-sideBar-border, #313244);
            --user-bg: var(--vscode-textBlockQuote-background, #45475a);
            --asst-bg: transparent;
            --error-bg: #f38ba820;
            --error-fg: #f38ba8;
            --accent: var(--vscode-focusBorder, #89b4fa);
            --input-bg: var(--vscode-input-background, #313244);
            --input-fg: var(--vscode-input-foreground, #cdd6f4);
            --radius: 8px;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: var(--vscode-font-family, -apple-system, sans-serif);
            font-size: 13px;
            background: var(--bg);
            color: var(--fg);
            display: flex;
            flex-direction: column;
            height: 100vh;
            overflow: hidden;
        }

        #header {
            padding: 10px 12px;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            gap: 8px;
            font-weight: 600;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            flex-shrink: 0;
        }

        #status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #a6adc8;
            flex-shrink: 0;
        }
        #status-dot.thinking { background: #f9e2af; animation: pulse 1s infinite; }
        #status-dot.idle { background: #a6e3a1; }
        #status-dot.error { background: #f38ba8; }
        #status-dot.disconnected { background: #6c7086; }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }

        #messages {
            flex: 1;
            overflow-y: auto;
            padding: 12px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .msg {
            padding: 10px 12px;
            border-radius: var(--radius);
            line-height: 1.5;
            word-break: break-word;
            white-space: pre-wrap;
            font-size: 13px;
        }

        .msg.user {
            background: var(--user-bg);
            align-self: flex-end;
            max-width: 92%;
        }

        .msg.assistant {
            background: var(--asst-bg);
            border-left: 3px solid var(--accent);
            padding-left: 10px;
        }

        .msg.error {
            background: var(--error-bg);
            color: var(--error-fg);
            border-left: 3px solid var(--error-fg);
            font-style: italic;
        }

        .msg.system {
            text-align: center;
            color: #6c7086;
            font-size: 11px;
            padding: 4px;
        }

        .msg .role-label {
            font-size: 10px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            opacity: 0.6;
            margin-bottom: 4px;
            display: block;
        }

        #typing-indicator {
            display: none;
            padding: 8px 12px;
            gap: 4px;
            align-items: center;
        }
        #typing-indicator.active { display: flex; }
        #typing-indicator span {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: #a6adc8;
            animation: bounce 1.4s infinite;
        }
        #typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
        #typing-indicator span:nth-child(3) { animation-delay: 0.4s; }

        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }

        #input-area {
            padding: 10px 12px;
            border-top: 1px solid var(--border);
            display: flex;
            gap: 8px;
            flex-shrink: 0;
        }

        #input-area input {
            flex: 1;
            background: var(--input-bg);
            color: var(--input-fg);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 8px 12px;
            font-size: 13px;
            outline: none;
        }
        #input-area input:focus {
            border-color: var(--accent);
        }

        #input-area button {
            background: var(--accent);
            color: var(--bg);
            border: none;
            border-radius: var(--radius);
            padding: 8px 14px;
            font-size: 13px;
            cursor: pointer;
            font-weight: 600;
        }
        #input-area button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .empty-state {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 8px;
            color: #6c7086;
            text-align: center;
            padding: 20px;
        }
        .empty-state .icon { font-size: 36px; }
        .empty-state .title { font-size: 14px; font-weight: 600; }
        .empty-state .hint { font-size: 12px; }

        code {
            font-family: var(--vscode-editor-font-family, monospace);
            font-size: 12px;
            background: var(--user-bg);
            padding: 1px 4px;
            border-radius: 4px;
        }

        pre {
            background: var(--input-bg);
            padding: 10px;
            border-radius: var(--radius);
            overflow-x: auto;
            margin: 6px 0;
        }
    </style>
</head>
<body>
    <div id="header">
        <div id="status-dot" class="disconnected"></div>
        <span>Hermes Agent</span>
    </div>

    <div id="messages">
        <div class="empty-state" id="empty-state">
            <div class="icon">🤖</div>
            <div class="title">Hermes Agent</div>
            <div class="hint">Select code and right-click for Explain, Fix, or Review.<br>Or type a message below to chat.</div>
        </div>
        <div id="typing-indicator">
            <span></span><span></span><span></span>
        </div>
    </div>

    <div id="input-area">
        <input type="text" id="message-input" placeholder="Ask Hermes something..."
               autofocus>
        <button id="send-btn" disabled>Send</button>
    </div>

    <script>
        const vscode = acquireVsCodeApi();
        const messagesEl = document.getElementById('messages');
        const emptyState = document.getElementById('empty-state');
        const typingIndicator = document.getElementById('typing-indicator');
        const statusDot = document.getElementById('status-dot');
        const input = document.getElementById('message-input');
        const sendBtn = document.getElementById('send-btn');
        let currentStatus = 'disconnected';

        // ── Handle messages from extension ─────────────────────
        window.addEventListener('message', event => {
            const msg = event.data;
            switch (msg.type) {
                case 'addMessage':
                    emptyState.style.display = 'none';
                    addMessageToDom(msg.message);
                    scrollToBottom();
                    break;
                case 'setStatus':
                    setStatus(msg.status);
                    break;
                case 'clear':
                    const allMsgs = messagesEl.querySelectorAll('.msg');
                    allMsgs.forEach(m => m.remove());
                    emptyState.style.display = 'flex';
                    break;
            }
        });

        function addMessageToDom(msg) {
            const div = document.createElement('div');
            div.className = 'msg ' + msg.role;
            div.id = msg.id;

            const label = document.createElement('span');
            label.className = 'role-label';
            const labels = { user: 'You', assistant: 'Hermes', error: 'Error', system: '' };
            label.textContent = labels[msg.role] || '';
            div.appendChild(label);

            const content = document.createElement('div');
            content.innerHTML = formatContent(msg.content);
            div.appendChild(content);

            // Insert before typing indicator
            messagesEl.insertBefore(div, typingIndicator);
        }

        function formatContent(text) {
            // Escape HTML
            let escaped = text
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;');

            // Code blocks (```code```)
            escaped = escaped.replace(/\x60\x60\x60([\\s\\S]*?)\x60\x60\x60/g,
                (_, code) => '<pre><code>' + code + '</code></pre>');

            // Inline code (`code`)
            escaped = escaped.replace(/\x60([^\x60]+)\x60/g,
                (_, code) => '<code>' + code + '</code>');

            // Bold (**text**)
            escaped = escaped.replace(/\\*\\*([^*]+)\\*\\*/g, '<strong>$1</strong>');

            // Line breaks
            escaped = escaped.replace(/\\n/g, '<br>');

            return escaped;
        }

        function setStatus(status) {
            currentStatus = status;
            statusDot.className = status;

            if (status === 'thinking') {
                typingIndicator.classList.add('active');
                sendBtn.disabled = true;
            } else {
                typingIndicator.classList.remove('active');
                sendBtn.disabled = false;
            }

            input.disabled = (status === 'thinking');
        }

        function scrollToBottom() {
            messagesEl.scrollTop = messagesEl.scrollHeight;
        }

        // ── Input handling ─────────────────────────────────────
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey && input.value.trim()) {
                e.preventDefault();
                sendMessage();
            }
        });

        input.addEventListener('input', () => {
            sendBtn.disabled = !input.value.trim() || currentStatus === 'thinking';
        });

        sendBtn.addEventListener('click', () => {
            if (input.value.trim()) {
                sendMessage();
            }
        });

        function sendMessage() {
            const text = input.value.trim();
            if (!text) return;

            // Add locally for instant feedback
            addMessageToDom({
                role: 'user',
                content: text,
                id: 'local-' + Date.now(),
                timestamp: Date.now(),
            });
            emptyState.style.display = 'none';
            scrollToBottom();

            // Send to extension
            vscode.postMessage({ type: 'sendMessage', text });
            input.value = '';
            sendBtn.disabled = true;
        }

        // Signal ready
        vscode.postMessage({ type: 'ready' });
    </script>
</body>
</html>`;
    }
}
