const cp = require('child_process');
const { EventEmitter } = require('events');
const readline = require('readline');

class AcpClient extends EventEmitter {
  constructor({ command = 'hermes', args = ['acp'], cwd = process.cwd(), env = process.env } = {}) {
    super();
    this.command = command;
    this.args = args;
    this.cwd = cwd;
    this.env = env;
    this.nextId = 1;
    this.pending = new Map();
    this.sessionId = undefined;
    this.proc = undefined;
    this.initialized = false;
    this.permissionHandler = undefined;
  }

  start() {
    if (this.proc) return;
    this.proc = cp.spawn(this.command, this.args, { cwd: this.cwd, env: this.env, stdio: ['pipe', 'pipe', 'pipe'] });
    this.proc.on('error', (error) => this._rejectAll(error));
    this.proc.on('close', (code) => {
      this.emit('status', `ACP exited with ${code}`);
      this._rejectAll(new Error(`ACP process exited with ${code}`));
      this.proc = undefined;
      this.initialized = false;
      this.sessionId = undefined;
    });
    this.proc.stderr.on('data', (chunk) => this.emit('stderr', chunk.toString()));
    const rl = readline.createInterface({ input: this.proc.stdout });
    rl.on('line', (line) => this._onLine(line));
  }

  async ensureSession(cwd) {
    this.start();
    if (!this.initialized) {
      await this.request('initialize', {
        protocolVersion: 1,
        clientInfo: { name: 'hermes-vscode', title: 'Hermes Code', version: '0.3.0' },
        clientCapabilities: {
          auth: { terminal: false },
          fs: { readTextFile: false, writeTextFile: false },
          terminal: false
        }
      });
      this.initialized = true;
    }
    if (!this.sessionId || this.cwd !== cwd) {
      this.cwd = cwd;
      const response = await this.request('session/new', { cwd, mcpServers: [] });
      this.sessionId = response.sessionId;
      this.emit('status', `ACP session ${this.sessionId}`);
    }
    return this.sessionId;
  }

  async prompt(text, cwd) {
    const sessionId = await this.ensureSession(cwd);
    const final = { text: '', thoughts: '', tools: [] };
    const onUpdate = (event) => {
      if (event.sessionId !== sessionId) return;
      const update = event.update || {};
      if (update.sessionUpdate === 'agent_message_chunk') {
        const textChunk = contentText(update.content);
        if (textChunk) {
          final.text += textChunk;
          this.emit('partial', final.text);
        }
      } else if (update.sessionUpdate === 'agent_thought_chunk') {
        const textChunk = contentText(update.content);
        if (textChunk) final.thoughts += textChunk;
      } else if (update.sessionUpdate === 'tool_call' || update.sessionUpdate === 'tool_call_update') {
        final.tools.push(update);
        this.emit('tool', update);
      } else if (update.sessionUpdate === 'plan') {
        this.emit('plan', update.entries || []);
      } else if (update.sessionUpdate === 'usage_update') {
        this.emit('usage', update);
      }
    };
    this.on('session_update', onUpdate);
    try {
      const response = await this.request('session/prompt', {
        sessionId,
        prompt: [{ type: 'text', text }]
      });
      return { ...final, response };
    } finally {
      this.off('session_update', onUpdate);
    }
  }

  async cancel() {
    if (this.sessionId) {
      await this.notify('session/cancel', { sessionId: this.sessionId });
    }
    if (this.proc) this.proc.kill('SIGTERM');
  }

  dispose() {
    if (this.proc) this.proc.kill('SIGTERM');
    this.proc = undefined;
    this._rejectAll(new Error('ACP client disposed'));
  }

  request(method, params) {
    const id = this.nextId++;
    const payload = { jsonrpc: '2.0', id, method, params };
    this._send(payload);
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, method });
    });
  }

  async notify(method, params) {
    this._send({ jsonrpc: '2.0', method, params });
  }

  _send(payload) {
    if (!this.proc || !this.proc.stdin.writable) throw new Error('ACP process is not running');
    this.proc.stdin.write(JSON.stringify(payload) + '\n');
  }

  _onLine(line) {
    if (!line.trim()) return;
    let message;
    try {
      message = JSON.parse(line);
    } catch (error) {
      this.emit('stderr', `Invalid ACP JSON: ${line}\n${error.message}`);
      return;
    }
    if (Object.prototype.hasOwnProperty.call(message, 'id') && !message.method) {
      const pending = this.pending.get(message.id);
      if (!pending) return;
      this.pending.delete(message.id);
      if (message.error) pending.reject(new Error(`${pending.method}: ${message.error.message || JSON.stringify(message.error)}`));
      else pending.resolve(message.result);
      return;
    }
    if (message.method && Object.prototype.hasOwnProperty.call(message, 'id')) {
      this._handleRequest(message);
      return;
    }
    if (message.method) this._handleNotification(message);
  }

  _handleNotification(message) {
    if (message.method === 'session/update') this.emit('session_update', message.params || {});
    else this.emit('notification', message);
  }

  async _handleRequest(message) {
    try {
      let result = null;
      if (message.method === 'session/request_permission') {
        result = await this._requestPermission(message.params || {});
      } else if (message.method === 'fs/read_text_file') {
        throw new Error('Hermes Code ACP client does not expose direct fs/read_text_file; use Hermes file tools instead.');
      } else if (message.method === 'fs/write_text_file') {
        throw new Error('Hermes Code ACP client does not expose direct fs/write_text_file; use patch preview/apply instead.');
      }
      this._send({ jsonrpc: '2.0', id: message.id, result });
    } catch (error) {
      this._send({ jsonrpc: '2.0', id: message.id, error: { code: -32603, message: error.message || String(error) } });
    }
  }

  async _requestPermission(params) {
    const options = params.options || [];
    this.emit('permission', params);
    if (this.permissionHandler) return this.permissionHandler(params);
    return { outcome: { outcome: 'cancelled' } };
  }

  _rejectAll(error) {
    for (const pending of this.pending.values()) pending.reject(error);
    this.pending.clear();
  }
}

function contentText(content) {
  if (!content) return '';
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) return content.map(contentText).join('');
  if (content.text) return content.text;
  if (content.content) return contentText(content.content);
  if (content.resource && content.resource.text) return content.resource.text;
  return '';
}

module.exports = { AcpClient, contentText };
