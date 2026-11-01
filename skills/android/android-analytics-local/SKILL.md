---
name: android-analytics-local
description: Add analytics to Android project without google-services.json using SharedPreferences + JSON local storage
category: android
tags: [analytics, android, mixpanel, firebase, local]
---

# Android Local Analytics

## Context
Adding analytics to an Android project without google-services.json (demo/dev environments).

## Problem
- Mixpanel Android SDK (`com.mixpanelandroid:mixpanel-android`) is NOT in standard MavenCentral — requires private repo token
- Firebase Analytics requires `google-services.json` configuration file

## Solution: LocalAnalytics

Use SharedPreferences + JSON to store events locally, with Timber logging for verification. Replace with Firebase/Mixpanel for production.

## Implementation

### Analytics interface
```kotlin
interface Analytics {
    fun track(event: String, properties: Map<String, Any>? = null)
    fun setUserProperty(key: String, value: Any)
    fun identify(userId: String)
    fun reset()
}
```

### LocalAnalytics
- Store events as JSONArray in SharedPreferences
- Include super properties: app_version, environment, timestamp
- Use Timber for debug logging
- Handle Map property type conversion (String/Int/Long/Double/Boolean)

### Hilt Module
```kotlin
@Module @InstallIn(SingletonComponent::class)
object AnalyticsModule {
    @Provides @Singleton
    fun provideAnalytics(@ApplicationContext context: Context): Analytics {
        return LocalAnalytics(context)
    }
}
```

## Common Events
| Event | Properties |
|-------|------------|
| `counter_increment` | `count: Int` |
| `counter_reset` | `previous_count: Int` |
| `theme_changed` | `previous_theme, new_theme: String` |
| `language_changed` | `previous_language, new_language: String` |
| `posts_loaded` | `count: Int` |
| `screen_view` | `screen: String` |

## Production Migration
1. Add `google-services.json` to `app/` directory
2. Add `classpath 'com.google.gms:google-services:4.4.0'` to project buildscript
3. Add `apply plugin: 'com.google.gms.google-services'` to app build.gradle
4. Replace LocalAnalytics with FirebaseAnalyticsService

## Build Notes
- No external dependencies needed
- Works completely offline
- Events visible in Timber logcat
- Events stored in SharedPreferences for debugging
