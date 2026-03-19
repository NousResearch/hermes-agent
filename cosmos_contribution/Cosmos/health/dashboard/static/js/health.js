/**
<<<<<<< HEAD:cosmos/health/dashboard/static/js/health.js
 * cosmos Health Dashboard JavaScript
=======
 * Cosmos Health Dashboard JavaScript
>>>>>>> dd5db7d5307d56ce54f13e61b92f95333530d4d1:cosmos/health/dashboard/static/js/health.js
 * Real-time health tracking and visualization
 */

// WebSocket connection
let healthWS = null;
let reconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 5;

// Chart instances
let charts = {};

// Initialize dashboard
function initHealthDashboard() {
    connectHealthWebSocket();
    setupEventListeners();
}

// WebSocket connection management
function connectHealthWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/health`;

    healthWS = new WebSocket(wsUrl);

    healthWS.onopen = () => {
        console.log('Health WebSocket connected');
        reconnectAttempts = 0;
        updateConnectionStatus(true);
    };

    healthWS.onclose = () => {
        console.log('Health WebSocket disconnected');
        updateConnectionStatus(false);
        attemptReconnect();
    };

    healthWS.onerror = (error) => {
        console.error('Health WebSocket error:', error);
    };

    healthWS.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleHealthMessage(data);
    };
}

function attemptReconnect() {
    if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
        reconnectAttempts++;
        const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000);
        console.log(`Reconnecting in ${delay}ms (attempt ${reconnectAttempts})`);
        setTimeout(connectHealthWebSocket, delay);
    }
}

function updateConnectionStatus(connected) {
    const statusDot = document.getElementById('wsStatus');
    const statusText = document.getElementById('connectionText');

    if (statusDot) {
        statusDot.style.background = connected ? '#00ff88' : '#ff6b6b';
    }
    if (statusText) {
        statusText.textContent = connected ? 'Connected' : 'Disconnected';
    }
}

// Handle incoming WebSocket messages
function handleHealthMessage(data) {
    switch (data.type) {
        case 'metric_update':
            updateMetricDisplay(data.data);
            break;
        case 'bio_data':
            updateBioData(data.data);
            break;
        case 'alert':
            showAlert(data.data);
            break;
        case 'heartbeat':
            // Keep-alive, no action needed
            break;
        default:
            console.log('Unknown message type:', data.type);
    }
}

// Update metric displays
function updateMetricDisplay(data) {
    const type = data.metric_type || data.signal_type;
    const value = data.value || data.processed_value;

    const metricElements = {
        'heart_rate': 'hrValue',
        'HR': 'hrValue',
        'steps': 'stepsValue',
        'STEPS': 'stepsValue',
        'heart_rate_variability': 'hrvValue',
        'HRV': 'hrvValue',
        'sleep_duration': 'sleepValue',
        'SLEEP_DURATION': 'sleepValue',
        'calories_burned': 'caloriesValue',
        'CALORIES_BURNED': 'caloriesValue',
        'recovery_score': 'recoveryValue',
        'RECOVERY_SCORE': 'recoveryValue',
    };

    const elementId = metricElements[type];
    if (elementId) {
        const element = document.getElementById(elementId);
        if (element) {
            element.textContent = formatMetricValue(type, value);
            animateUpdate(element);
        }
    }

    // Update charts if applicable
    updateChartData(type, value);
}

function updateBioData(data) {
    // Same as metric update for bio data packets
    updateMetricDisplay({
        metric_type: data.signal_type,
        value: data.value
    });
}

// Format metric values for display
function formatMetricValue(type, value) {
    switch (type) {
        case 'heart_rate':
        case 'HR':
        case 'heart_rate_variability':
        case 'HRV':
        case 'recovery_score':
        case 'RECOVERY_SCORE':
            return Math.round(value);
        case 'steps':
        case 'STEPS':
        case 'calories_burned':
        case 'CALORIES_BURNED':
            return Math.round(value).toLocaleString();
        case 'sleep_duration':
        case 'SLEEP_DURATION':
            return value.toFixed(1);
        default:
            return typeof value === 'number' ? value.toFixed(1) : value;
    }
}

// Animate value updates
function animateUpdate(element) {
    element.classList.add('animate-pulse');
    setTimeout(() => {
        element.classList.remove('animate-pulse');
    }, 500);
}

// Update chart data
function updateChartData(type, value) {
    if (type === 'HR' || type === 'heart_rate') {
        const chart = charts['hrChart'];
        if (chart) {
            chart.data.datasets[0].data.push(value);
            chart.data.datasets[0].data.shift();
            chart.update('none');
        }
    }
}

// Show alerts
function showAlert(alertData) {
    const alertsContainer = document.getElementById('alertsList');
    if (!alertsContainer) return;

    const alertElement = document.createElement('div');
    alertElement.className = 'alert-item animate-slide-in';
    alertElement.innerHTML = `
        <div class="alert-icon ${alertData.severity}">
            ${alertData.severity === 'critical' ? '&#9888;' :
              alertData.severity === 'warning' ? '&#9888;' : '&#8505;'}
        </div>
        <div class="alert-content">
            <div class="alert-title">${alertData.title}</div>
            <div class="alert-message">${alertData.message}</div>
        </div>
    `;

    alertsContainer.prepend(alertElement);

    // Auto-remove after 30 seconds
    setTimeout(() => {
        alertElement.remove();
    }, 30000);
}

// API helper functions
async function fetchHealthData(endpoint, options = {}) {
    try {
        const response = await fetch(endpoint, {
            headers: {
                'Content-Type': 'application/json',
            },
            ...options,
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error(`Error fetching ${endpoint}:`, error);
        throw error;
    }
}

async function postHealthData(endpoint, data) {
    return fetchHealthData(endpoint, {
        method: 'POST',
        body: JSON.stringify(data),
    });
}

// Common data loading functions
async function loadDailySummary(dateStr = null) {
    const url = dateStr ? `/api/health/summary?date_str=${dateStr}` : '/api/health/summary';
    return fetchHealthData(url);
}

async function loadMetrics(metricType, startDate, endDate) {
    let url = `/api/health/metrics/${metricType}`;
    const params = new URLSearchParams();
    if (startDate) params.append('start_date', startDate);
    if (endDate) params.append('end_date', endDate);
    if (params.toString()) url += `?${params}`;
    return fetchHealthData(url);
}

async function loadTrends(days = 7) {
    return fetchHealthData(`/api/health/trends?days=${days}`);
}

async function loadAlerts() {
    return fetchHealthData('/api/health/alerts');
}

async function loadInsights() {
    return fetchHealthData('/api/health/insights');
}

async function loadGoals() {
    return fetchHealthData('/api/goals');
}

async function createGoal(goalData) {
    return postHealthData('/api/goals', goalData);
}

async function updateGoal(goalId, updates) {
    return fetchHealthData(`/api/goals/${goalId}`, {
        method: 'PUT',
        body: JSON.stringify(updates),
    });
}

// Nutrition functions
async function loadDailyNutrition(dateStr = null) {
    const url = dateStr ? `/api/nutrition/daily?date_str=${dateStr}` : '/api/nutrition/daily';
    return fetchHealthData(url);
}

async function searchFoods(query, limit = 20) {
    return fetchHealthData(`/api/nutrition/foods?query=${encodeURIComponent(query)}&limit=${limit}`);
}

async function logMeal(mealData) {
    return postHealthData('/api/nutrition/meals', mealData);
}

// Document parsing
async function parseDocument(file, docType) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('doc_type', docType);

    const response = await fetch('/api/documents/parse', {
        method: 'POST',
        body: formData,
    });

    return response.json();
}

// Utility functions
function formatDate(date) {
    return date.toISOString().split('T')[0];
}

function formatTime(date) {
    return date.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit'
    });
}

function formatDateTime(date) {
    return `${formatDate(date)} ${formatTime(date)}`;
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function throttle(func, limit) {
    let inThrottle;
    return function(...args) {
        if (!inThrottle) {
            func.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// Event listeners setup
function setupEventListeners() {
    // Ping server every 25 seconds to keep connection alive
    setInterval(() => {
        if (healthWS && healthWS.readyState === WebSocket.OPEN) {
            healthWS.send(JSON.stringify({ type: 'ping' }));
        }
    }, 25000);

    // Visibility change handler
    document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'visible') {
            if (!healthWS || healthWS.readyState !== WebSocket.OPEN) {
                connectHealthWebSocket();
            }
        }
    });
}

// Export functions for use in templates
window.HealthDashboard = {
    init: initHealthDashboard,
    loadDailySummary,
    loadMetrics,
    loadTrends,
    loadAlerts,
    loadInsights,
    loadGoals,
    createGoal,
    updateGoal,
    loadDailyNutrition,
    searchFoods,
    logMeal,
    parseDocument,
    formatDate,
    formatTime,
    formatDateTime,
    debounce,
    throttle,
};
