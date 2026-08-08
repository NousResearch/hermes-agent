/**
 * Hermes Agent VS Code Extension — Main Entry Point
 *
 * Activates on startup, registers commands, creates the chat sidebar,
 * and manages the MCP client lifecycle.
 */

import * as vscode from 'vscode';
import { HermesMCPClient } from './mcpClient';
import { ChatPanel } from './chatPanel';

let client: HermesMCPClient;
let chatPanel: ChatPanel | undefined;

export function activate(context: vscode.ExtensionContext) {
    const config = vscode.workspace.getConfiguration('hermes');
    const outputChannel = vscode.window.createOutputChannel('Hermes Agent', { log: true });
    outputChannel.appendLine('Hermes Agent extension activating...');

    // ── MCP Client ─────────────────────────────────────────────
    client = new HermesMCPClient({
        command: config.get<string>('mcpCommand', 'hermes'),
        args: config.get<string[]>('mcpArgs', ['mcp', 'serve']),
        outputChannel,
        autoConnect: config.get<boolean>('autoConnect', true),
    });

    // ── Chat Panel ─────────────────────────────────────────────
    chatPanel = new ChatPanel(context.extensionUri, outputChannel);

    // ── Register Commands ──────────────────────────────────────
    context.subscriptions.push(
        vscode.commands.registerCommand('hermes.showChat', () => {
            chatPanel?.reveal();
        }),

        vscode.commands.registerCommand('hermes.explainCode', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) { return; }
            const selection = editor.document.getText(editor.selection);
            const language = editor.document.languageId;
            const fileName = editor.document.fileName.split('/').pop() || '';
            await sendTask(`Explain the following ${language} code from ${fileName}:\n\`\`\`${language}\n${selection}\n\`\`\``);
        }),

        vscode.commands.registerCommand('hermes.fixCode', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) { return; }
            const selection = editor.document.getText(editor.selection);
            const language = editor.document.languageId;
            await sendTask(`Fix any bugs or issues in this ${language} code. Return the corrected code:\n\`\`\`${language}\n${selection}\n\`\`\``);
        }),

        vscode.commands.registerCommand('hermes.reviewCode', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) { return; }
            const selection = editor.document.getText(editor.selection);
            const language = editor.document.languageId;
            await sendTask(`Review this ${language} code for bugs, security issues, and improvement opportunities:\n\`\`\`${language}\n${selection}\n\`\`\``);
        }),

        vscode.commands.registerCommand('hermes.customTask', async () => {
            const editor = vscode.window.activeTextEditor;
            const task = await vscode.window.showInputBox({
                prompt: 'What would you like Hermes to do?',
                placeHolder: 'e.g. Add error handling to this function...',
            });
            if (!task) { return; }
            if (editor && !editor.selection.isEmpty) {
                const selection = editor.document.getText(editor.selection);
                const language = editor.document.languageId;
                await sendTask(`${task}\n\nContext:\n\`\`\`${language}\n${selection}\n\`\`\``);
            } else {
                await sendTask(task);
            }
        }),

        vscode.commands.registerCommand('hermes.sendSelection', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) { return; }
            const selection = editor.document.getText(editor.selection);
            if (selection) {
                chatPanel?.addUserMessage(selection);
                await client.sendMessage(selection);
            }
        }),

        vscode.commands.registerCommand('hermes.stopAgent', () => {
            client.cancelCurrent();
            chatPanel?.setStatus('idle');
            vscode.window.showInformationMessage('Hermes: stopped current task.');
        }),

        vscode.commands.registerCommand('hermes.clearChat', () => {
            chatPanel?.clear();
        }),
    );

    // ── Model selection via config change ──────────────────────
    context.subscriptions.push(
        vscode.workspace.onDidChangeConfiguration(e => {
            if (e.affectsConfiguration('hermes.model')) {
                const model = vscode.workspace.getConfiguration('hermes').get<string>('model', '');
                outputChannel.appendLine(`Model changed to: ${model || 'default'}`);
            }
        }),
    );

    // ── Auto-connect ───────────────────────────────────────────
    if (config.get<boolean>('autoConnect', true)) {
        client.connect().catch(err => {
            outputChannel.appendLine(`Auto-connect failed: ${err.message}`);
        });
    }

    outputChannel.appendLine('Hermes Agent extension activated.');
}

export function deactivate() {
    client?.disconnect();
}

async function sendTask(prompt: string): Promise<void> {
    if (!client.isConnected()) {
        const connect = await vscode.window.showInformationMessage(
            'Hermes is not connected. Connect now?',
            'Connect', 'Cancel'
        );
        if (connect !== 'Connect') { return; }
        await client.connect();
    }

    chatPanel?.reveal();
    chatPanel?.addUserMessage(prompt);
    chatPanel?.setStatus('thinking');

    try {
        const response = await client.sendMessage(prompt);
        chatPanel?.addAssistantMessage(response);
        chatPanel?.setStatus('idle');
    } catch (err: any) {
        chatPanel?.addErrorMessage(err.message || 'Unknown error');
        chatPanel?.setStatus('error');
    }
}
