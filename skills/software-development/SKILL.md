---
name: android-hello-iteration-workflow
description: PRJ-20260417-002 Android 应用高速迭代流程 — MVVM + Compose + Hilt + Room + WorkManager。PRD确认后小墨自行决定推进。
tags: [android, kotlin, gradle, compose, hilt, room, workmanager, datastore]
---

# android-hello Iteration Workflow

PRJ-20260417-002 Android 应用高速迭代流程 — MVVM + Compose + Hilt + Room + WorkManager。

## 适用场景
android-hello 项目从 V2.0.0 到 V3.x 的功能迭代。PRD确认后小墨自行决定推进。

## 版本历史
| 版本 | 功能 | 状态 |
|------|------|------|
| V2.0.0 | MVVM + Compose + Hilt | delivered |
| V2.1.0 | Unit Tests | delivered |
| V2.2.0 | Settings + BottomNavigation | delivered |
| V2.3.0 | Room 持久化 | delivered |
| V2.4.0 | Retrofit 网络层 | delivered |
| V2.5.0 | SavedStateHandle | delivered |
| V2.6.0 | 深色模式 | delivered |
| V2.7.0 | Compose UI 测试 | delivered |
| V2.8.0 | 国际化 (中/英) | delivered |
| V2.9.0 | CI/CD + APK 发布 | delivered |
| V3.0.0 | App Shortcuts + Widget | delivered |
| V3.1.0 | Push Notifications | delivered |
| V3.2.0 | Analytics | delivered |
| V3.3.0 | WorkManager 后台任务 | delivered |
| V3.4.0 | DataStore 替代 SharedPreferences | delivered |
| V3.5.0 | R8 性能优化 | delivered |

## 迭代流程
1. boss 选择方向 (A/B/C/D/E) → 小墨 PRD 起草
2. boss 确认 → 小墨自行实现
3. Gradle 构建 (assembleDevDebug)
4. git add → git commit → git push
5. 更新 proposals.csv 和 project-index.md
6. 交付报告

## 关键实现要点

### Widget 实现 (V3.0.0)
- 使用 AppWidgetProvider + PendingIntent（不用 Glance，简单）
- CounterWidgetReceiver 处理 widget 点击 ACTION_INCREMENT
- SharedPreferences 持久化（后续迁移到 DataStore V3.4.0）
- MainViewModel.syncWithWidgetCounter() 启动时同步

### Notification 实现 (V3.1.0)
- NotificationCompat + NotificationChannel (Android 8.0+)
- POST_NOTIFICATIONS 权限 (Android 13+)
- NotificationHelper.createNotificationChannel() 在 HelloApp.onCreate() 调用
- 计数器里程碑通知：count % 10 == 0 时触发

### Analytics 实现 (V3.2.0)
- Mixpanel/Firebase SDK 需要 google-services.json，android-hello 未接入
- 使用 LocalAnalytics 替代：SharedPreferences JSON 存储事件
- Analytics 接口：track(), setUserProperty(), identify(), reset()
- AnalyticsModule 通过 Hilt 单例注入

### WorkManager 实现 (V3.3.0)
- HiltWorker 依赖：hilt-work:1.2.0 + hilt-compiler:1.2.0
- HelloApp 实现 Configuration.Provider 接口
- AndroidManifest 必须移除默认 InitializationProvider：
  ```xml
  <provider
      android:name="androidx.startup.InitializationProvider"
      android:authorities="${applicationId}.androidx-startup"
      android:exported="false"
      tools:node="remove" />
  ```
- 15分钟定时刷新，网络连接约束

### DataStore 实现 (V3.4.0)
- datastore-preferences:1.0.0
- CounterDataStore: counterFlow (Flow<Int>), incrementCounter(), setCounter(), resetCounter()
- CounterWidgetReceiver 使用 DataStore（AppWidgetProvider 中直接实例化 DataStore(context)）
- MainViewModel 通过 Hilt 注入 CounterDataStore

### R8 性能优化 (V3.5.0)
- minifyEnabled true + shrinkResources true
- proguard-rules.pro 覆盖 Hilt/Room/Retrofit/Coroutines/DataStore/Compose
- Lint 修复：必须移除 WorkManager 默认 initializer（见上方）
- 压缩效果：26M → 8.2M (68% reduction)

## Build 命令
```bash
# Debug build
gradle assembleDevDebug --no-daemon

# Release build (R8 minified)
gradle assembleProdRelease --no-daemon

# 构建日志
tail -40 /tmp/dev-debug-build.log
```

## 常见坑点
1. **WorkManager Lint 错误**：使用 Configuration.Provider 时必须移除 AndroidManifest 中的默认 initializer
2. **AppWidgetProvider context**：不能在类级别引用 context，要在方法参数或回调中获取
3. **Mixpanel/Firebase SDK**：需要 google-services.json，未接入时用 LocalAnalytics 替代
4. **R8 移除关键代码**：proguard-rules.pro 必须保留 Hilt、Room、Parcelable 相关 keep 规则

## APK 发布与签名

### Release 构建签名配置
```groovy
// app/build.gradle
android {
    signingConfigs {
        release {
            storeFile file("${System.getProperty("user.home")}/.android/debug.keystore")
            storePassword "android"
            keyAlias "androiddebugkey"
            keyPassword "android"
        }
    }
    buildTypes {
        release {
            signingConfig signingConfigs.release  // 或 signingConfig signingConfigs.debug
            minifyEnabled false
        }
    }
}
```

### 后签名方案（构建时未签名）
```bash
# 1. 生成 debug keystore（如果不存在）
keytool -genkeypair -keystore debug.keystore -storepass android \
    -alias androiddebugkey -keypass android -keyalg RSA -keysize 2048 \
    -validity 10000 -dname "CN=Android Debug,O=Android,C=US"

# 2. jarsigner 签名（v1 签名，APK 必须）
jarsigner -keystore debug.keystore -storepass android \
    -keypass android -signedjar output-signed.apk \
    input-unsigned.apk androiddebugkey
```

### APK 损坏诊断
- **症状**：安装时报"安装包已损坏"
- **检查**：
  ```bash
  file app-release.apk                              # 应显示 "Android package"
  unzip -l app-release.apk | head -20               # 应列出 ZIP entries
  xxd app-release.apk | head -2                     # 开头应是 50 4b 03 04
  ```
- **常见原因**：
  - 下载中断导致文件截断（开头不是 PK header）
  - 文件大小与 GitHub 显示不匹配（GitHub 显示 13MB，本地只有 1MB）
  - 缺少签名（MANIFEST.MF、CERT.RSA/CERT.SF 文件不存在）

### GitHub Release 上传
```bash
# 上传 APK 到 Release
gh release upload v3.6.0 app-prod-release.apk --repo YeLuo45/android-hello

# 查看 Release assets
gh release view v3.6.0 --repo YeLuo45/android-hello --json assets

# 删除损坏的 asset
gh api -X DELETE repos/YeLuo45/android-hello/releases/assets/ASSET_ID
```

### Gradle Wrapper 损坏处理
- **症状**：`gradlew` 报 `.zip END header not found`
- **原因**：gradle-*-bin.zip 下载中断，留下 .zip.bad 或 .zip.part 文件
- **解决**：删除整个 wrapper 目录让其重新下载
  ```bash
  rm -rf ~/.gradle/wrapper/dists/gradle-8.10.2-bin/
  # 或使用系统 gradle
  export PATH=$HOME/.local/bin:$PATH
  gradle assembleProdRelease
  ```

## V3.6.0 Splash Screen API
- 当前版本：v3.6.0
- 2026-05-08：修复 APK 签名问题，重新构建并上传
