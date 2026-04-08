document.addEventListener('DOMContentLoaded', () => {
    // State management
    const state = {
        currentLanguage: 'en',
        currentTheme: 'dark',
        translations: {},
        sessionId: null,
        isThinking: false
    };

    // UI Elements
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const welcomeBanner = document.querySelector('.welcome-banner');
    const statusDot = document.querySelector('.status-dot');
    const statusText = document.getElementById('status-text');
    const currentModelDisplay = document.getElementById('current-model');
    const sessionsList = document.getElementById('sessions-list');
    const settingsBtn = document.getElementById('settings-btn');
    const settingsModal = document.getElementById('settings-modal');
    const closeModal = document.querySelector('.close-modal');
    const themeToggles = document.querySelectorAll('.theme-toggle');
    const langSelect = document.getElementById('lang-select');
    const modelInput = document.getElementById('model-input');
    const saveSettingsBtn = document.getElementById('save-settings');
    const newChatBtn = document.getElementById('new-chat-btn');

    // 1. I18n Implementation
    async function loadTranslations(lang) {
        try {
            const response = await fetch(`/static/locales/${lang}.json`);
            state.translations = await response.json();
            state.currentLanguage = lang;
            applyTranslations();
        } catch (error) {
            console.error('Error loading translations:', error);
        }
    }

    function applyTranslations() {
        // Elements with data-i18n attribute
        document.querySelectorAll('[data-i18n]').forEach(el => {
            const key = el.getAttribute('data-i18n');
            if (state.translations[key]) {
                el.textContent = state.translations[key];
            }
        });

        // Placeholder translations
        document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
            const key = el.getAttribute('data-i18n-placeholder');
            if (state.translations[key]) {
                el.placeholder = state.translations[key];
            }
        });

        // Tooltip translations
        document.querySelectorAll('[data-tooltip]').forEach(el => {
            const key = el.getAttribute('data-tooltip');
            if (state.translations[key]) {
                el.setAttribute('data-tooltip-text', state.translations[key]);
            }
        });

        // Update document title
        if (state.translations.app_title) {
            document.title = state.translations.app_title;
        }
    }

    // 2. Theme Implementation
    function setTheme(theme) {
        state.currentTheme = theme;
        if (theme === 'light') {
            document.body.classList.remove('dark-theme');
            document.body.classList.add('light-theme');
        } else {
            document.body.classList.remove('light-theme');
            document.body.classList.add('dark-theme');
        }

        themeToggles.forEach(btn => {
            btn.classList.toggle('active', btn.getAttribute('data-theme') === theme);
        });

        localStorage.setItem('hermes_theme', theme);
    }

    // 3. Chat Implementation
    function addMessage(role, content) {
        if (welcomeBanner) welcomeBanner.style.display = 'none';

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        messageDiv.innerHTML = `<div class="message-content">${formatContent(content)}</div>`;
        chatMessages.appendChild(messageDiv);

        // Auto-scroll to bottom
        const container = document.getElementById('chat-container');
        container.scrollTop = container.scrollHeight;

        return messageDiv;
    }

    function escapeHTML(str) {
        const p = document.createElement('p');
        p.textContent = str;
        return p.innerHTML;
    }

    function formatContent(content) {
        // Basic markdown-like formatting (could use a library like marked.js)
        if (!content) return '';

        // Escape HTML first for security
        let escaped = escapeHTML(content);

        return escaped
            .replace(/\n/g, '<br>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>');
    }

    async function sendMessage() {
        const message = userInput.value.trim();
        if (!message || state.isThinking) return;

        userInput.value = '';
        addMessage('user', message);
        setThinking(true);

        const url = `/api/chat/stream?message=${encodeURIComponent(message)}${state.sessionId ? `&session_id=${state.sessionId}` : ''}`;
        const eventSource = new EventSource(url);

        let assistantMessageDiv = null;
        let assistantContent = '';

        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.type === 'start') {
                state.sessionId = data.session_id;
                assistantMessageDiv = addMessage('assistant', '');
            } else if (data.type === 'delta') {
                assistantContent += data.content;
                assistantMessageDiv.querySelector('.message-content').innerHTML = formatContent(assistantContent);
                const container = document.getElementById('chat-container');
                container.scrollTop = container.scrollHeight;
            } else if (data.type === 'done') {
                eventSource.close();
                setThinking(false);
                loadSessions(); // Refresh sessions list
            } else if (data.type === 'error') {
                eventSource.close();
                setThinking(false);
                addMessage('assistant', `Error: ${data.message}`);
            }
        };

        eventSource.onerror = () => {
            eventSource.close();
            setThinking(false);
        };
    }

    function setThinking(isThinking) {
        state.isThinking = isThinking;
        statusDot.classList.toggle('thinking', isThinking);
        const key = isThinking ? 'status_thinking' : 'status_online';
        if (state.translations[key]) {
            statusText.textContent = state.translations[key];
        }
    }

    // 4. Session & Config Management
    async function loadSessions() {
        try {
            const response = await fetch('/api/sessions');
            const sessions = await response.json();

            sessionsList.innerHTML = '';
            if (sessions.length === 0) {
                const li = document.createElement('li');
                li.setAttribute('data-i18n', 'no_sessions');
                li.textContent = state.translations.no_sessions || 'No sessions found.';
                sessionsList.appendChild(li);
                return;
            }

            sessions.forEach(session => {
                const li = document.createElement('li');
                li.textContent = session.title || session.preview || session.session_id.substring(0, 15);
                li.title = session.session_id;
                li.onclick = () => loadSessionHistory(session.session_id);
                sessionsList.appendChild(li);
            });
        } catch (error) {
            console.error('Error loading sessions:', error);
        }
    }

    async function loadSessionHistory(sessionId) {
        try {
            const response = await fetch(`/api/sessions/${sessionId}`);
            if (!response.ok) throw new Error('Failed to load session');

            const data = await response.json();
            state.sessionId = sessionId;
            chatMessages.innerHTML = '';

            if (data.history && data.history.length > 0) {
                data.history.forEach(msg => {
                    // map role 'human' or 'user' to 'user', 'ai' or 'assistant' to 'assistant'
                    let role = 'user';
                    if (msg.role === 'assistant' || msg.role === 'ai') {
                        role = 'assistant';
                    } else if (msg.role === 'system') {
                        return; // Skip system messages in UI
                    }
                    addMessage(role, msg.content);
                });
            } else {
                addMessage('assistant', `Resumed empty session ${sessionId}`);
            }
        } catch (error) {
            console.error('Error loading session history:', error);
            addMessage('assistant', `Error loading session: ${error.message}`);
        }
    }

    async function loadConfig() {
        try {
            const response = await fetch('/api/config');
            const config = await response.json();

            currentModelDisplay.textContent = config.model || 'Default Model';
            modelInput.value = config.model || '';
            langSelect.value = config.language || 'en';

            // Apply loaded config
            await loadTranslations(config.language || 'en');
            setTheme(config.theme || 'dark');
        } catch (error) {
            console.error('Error loading config:', error);
            // Default fallbacks
            loadTranslations('en');
        }
    }

    async function saveSettings() {
        const update = {
            model: modelInput.value,
            language: langSelect.value,
            theme: state.currentTheme
        };

        try {
            const response = await fetch('/api/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(update)
            });

            if (response.ok) {
                currentModelDisplay.textContent = update.model;
                if (update.language !== state.currentLanguage) {
                    await loadTranslations(update.language);
                }
                settingsModal.classList.remove('open');
            }
        } catch (error) {
            console.error('Error saving config:', error);
        }
    }

    // Event Listeners
    sendBtn.onclick = sendMessage;
    userInput.onkeydown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    };

    newChatBtn.onclick = () => {
        state.sessionId = null;
        chatMessages.innerHTML = '';
        if (welcomeBanner) welcomeBanner.style.display = 'block';
    };

    settingsBtn.onclick = () => settingsModal.classList.add('open');
    closeModal.onclick = () => settingsModal.classList.remove('open');
    window.onclick = (e) => {
        if (e.target === settingsModal) settingsModal.classList.remove('open');
    };

    themeToggles.forEach(btn => {
        btn.onclick = () => setTheme(btn.getAttribute('data-theme'));
    });

    saveSettingsBtn.onclick = saveSettings;

    // Initialization
    loadConfig();
    loadSessions();
});
