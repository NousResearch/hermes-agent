# Android ProGuard rules
-keep class com.nousresearch.hermes.** { *; }

# Keep all activities and services
-keep public class * extends android.app.Activity
-keep public class * extends android.app.Service
-keep public class * extends android.content.BroadcastReceiver

# Keep all views
-keep public class * extends android.view.View {
    public <init>(android.content.Context);
    public <init>(android.content.Context, android.util.AttributeSet);
    public <init>(android.content.Context, android.util.AttributeSet, int);
}

# Keep R classes
-keep class **.R$* { *; }

# Keep Kotlin metadata
-keep class **.*$* {
    *;
}

# Keep Chaquopy Python classes
-keep class com.chaquo.python.** { *; }
-keep class org.python.** { *; }

# Keep native methods
-keep class * extends java.lang.Object {
    native <methods>;
}
