package com.mobilefork.hermesagent.device

import android.annotation.SuppressLint
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.graphics.Color
import android.graphics.PixelFormat
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Build
import android.os.IBinder
import android.provider.Settings
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.WindowManager
import android.widget.TextView
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import com.mobilefork.hermesagent.MainActivity
import com.mobilefork.hermesagent.R
import com.mobilefork.hermesagent.data.DeviceCapabilityStore
import kotlin.math.abs
import kotlin.math.max

class HermesFloatingButtonService : Service() {
    private var manager: WindowManager? = null
    private var floatingView: View? = null

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
        promoteToForeground()
        running = true
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (!Settings.canDrawOverlays(applicationContext)) {
            lastStartError = "Android overlay permission is not granted"
            stopSelf()
            return START_NOT_STICKY
        }
        running = true
        promoteToForeground()
        runCatching {
            showButton()
        }.onFailure { error ->
            lastStartError = error.message ?: error.javaClass.simpleName
            buttonVisible = false
            stopSelf()
            return START_NOT_STICKY
        }
        DeviceStateWriter.write(applicationContext)
        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        removeButton()
        running = false
        DeviceStateWriter.write(applicationContext)
        super.onDestroy()
    }

    private fun promoteToForeground() {
        val notification = buildNotification()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
    }

    private fun buildNotification(): Notification {
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_nav_hermes)
            .setContentTitle("Hermes floating button")
            .setContentText("Hermes stays available from Home and other apps")
            .setContentIntent(openAppPendingIntent())
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()
    }

    private fun openAppPendingIntent(): PendingIntent {
        val intent = Intent(this, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP
        }
        return PendingIntent.getActivity(
            this,
            0,
            intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) {
            return
        }
        val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        if (notificationManager.getNotificationChannel(CHANNEL_ID) != null) {
            return
        }
        val channel = NotificationChannel(
            CHANNEL_ID,
            "Hermes floating button",
            NotificationManager.IMPORTANCE_LOW,
        ).apply {
            description = "Keeps a floating Hermes launcher button above other apps"
        }
        notificationManager.createNotificationChannel(channel)
    }

    private fun showButton() {
        val windowManager = manager ?: (applicationContext.getSystemService(Context.WINDOW_SERVICE) as WindowManager).also {
            manager = it
        }
        if (floatingView != null) {
            return
        }
        val sizePx = dp(56)
        val layoutParams = initialLayoutParams(sizePx)
        val button = buildButton(windowManager, layoutParams)
        windowManager.addView(button, layoutParams)
        floatingView = button
        buttonVisible = true
        lastStartError = ""
    }

    private fun removeButton() {
        val view = floatingView ?: return
        floatingView = null
        buttonVisible = false
        runCatching {
            manager?.removeView(view)
        }
    }

    @SuppressLint("ClickableViewAccessibility")
    private fun buildButton(windowManager: WindowManager, layoutParams: WindowManager.LayoutParams): TextView {
        val density = resources.displayMetrics.density.takeIf { it > 0f } ?: 1f
        var downRawX = 0f
        var downRawY = 0f
        var startX = 0
        var startY = 0
        return TextView(this).apply {
            text = "H"
            textSize = 22f
            typeface = Typeface.DEFAULT_BOLD
            setTextColor(Color.WHITE)
            gravity = Gravity.CENTER
            contentDescription = "Open Hermes"
            importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_YES
            elevation = 14f * density
            background = GradientDrawable(
                GradientDrawable.Orientation.TL_BR,
                intArrayOf(0xFF5D5FEF.toInt(), 0xFF19C6A7.toInt()),
            ).apply {
                shape = GradientDrawable.OVAL
                setStroke(dp(2), 0xFFFFFFFF.toInt())
            }
            setOnClickListener {
                openHermes()
            }
            setOnTouchListener { view, event ->
                when (event.actionMasked) {
                    MotionEvent.ACTION_DOWN -> {
                        downRawX = event.rawX
                        downRawY = event.rawY
                        startX = layoutParams.x
                        startY = layoutParams.y
                        true
                    }
                    MotionEvent.ACTION_MOVE -> {
                        layoutParams.x = (startX + (event.rawX - downRawX)).toInt().coerceIn(
                            0,
                            max(0, screenWidthPx() - view.width),
                        )
                        layoutParams.y = max(0, startY + (event.rawY - downRawY).toInt())
                        runCatching { windowManager.updateViewLayout(view, layoutParams) }
                        true
                    }
                    MotionEvent.ACTION_UP -> {
                        if (abs(event.rawX - downRawX) < dp(8) && abs(event.rawY - downRawY) < dp(8)) {
                            view.performClick()
                        }
                        true
                    }
                    else -> false
                }
            }
        }
    }

    private fun openHermes() {
        val intent = Intent(this, MainActivity::class.java).apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP)
            putExtra(EXTRA_FROM_FLOATING_BUTTON, true)
        }
        startActivity(intent)
    }

    private fun initialLayoutParams(sizePx: Int): WindowManager.LayoutParams {
        val edge = dp(16)
        return WindowManager.LayoutParams(
            sizePx,
            sizePx,
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
            } else {
                @Suppress("DEPRECATION")
                WindowManager.LayoutParams.TYPE_PHONE
            },
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
            PixelFormat.TRANSLUCENT,
        ).apply {
            gravity = Gravity.TOP or Gravity.START
            x = max(edge, screenWidthPx() - sizePx - edge)
            y = dp(156)
            title = "Hermes floating button"
        }
    }

    private fun screenWidthPx(): Int {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            runCatching {
                val bounds = requireNotNull(manager).currentWindowMetrics.bounds
                if (bounds.width() > 0) {
                    return bounds.width()
                }
            }
        }
        @Suppress("DEPRECATION")
        return resources.displayMetrics.widthPixels
    }

    private fun dp(value: Int): Int {
        val density = resources.displayMetrics.density.takeIf { it > 0f } ?: 1f
        return (value * density).toInt()
    }

    companion object {
        private const val CHANNEL_ID = "hermes_floating_button"
        private const val NOTIFICATION_ID = 7319
        private const val EXTRA_FROM_FLOATING_BUTTON = "hermes_from_floating_button"

        @Volatile
        private var running: Boolean = false
        @Volatile
        private var buttonVisible: Boolean = false
        @Volatile
        private var lastStartError: String = ""

        fun start(context: Context): Boolean {
            val appContext = context.applicationContext
            if (!Settings.canDrawOverlays(appContext)) {
                lastStartError = "Android overlay permission is not granted"
                return false
            }
            val intent = Intent(appContext, HermesFloatingButtonService::class.java)
            return runCatching {
                ContextCompat.startForegroundService(appContext, intent)
                running = true
                true
            }.getOrElse { error ->
                lastStartError = error.message ?: error.javaClass.simpleName
                running = false
                buttonVisible = false
                false
            }
        }

        fun startIfDesired(context: Context): Boolean {
            val appContext = context.applicationContext
            if (!DeviceCapabilityStore(appContext).load().floatingButtonEnabled) {
                return false
            }
            return start(appContext)
        }

        fun stop(context: Context) {
            running = false
            buttonVisible = false
            lastStartError = ""
            context.applicationContext.stopService(Intent(context.applicationContext, HermesFloatingButtonService::class.java))
        }

        fun isRunning(): Boolean = running
        fun isButtonVisible(): Boolean = buttonVisible
        fun lastError(): String = lastStartError
    }
}
