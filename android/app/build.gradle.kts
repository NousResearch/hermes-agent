import java.util.Properties

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
    id("com.chaquo.python")
}

val repoRoot = rootDir.parentFile
val hermesVersionFile = repoRoot.resolve("hermes_cli/__init__.py")
val releaseTag = System.getenv("HERMES_RELEASE_TAG").orEmpty().trim()
val keystorePropertiesFile = rootDir.resolve("keystore.properties")
val keystoreProperties = Properties().apply {
    if (keystorePropertiesFile.isFile) {
        keystorePropertiesFile.inputStream().use(::load)
    }
}
val hasReleaseKeystore = keystoreProperties.isNotEmpty()

fun hermesVersionName(): String {
    val text = hermesVersionFile.readText()
    val match = Regex("""__version__\s*=\s*\"([^\"]+)\"""").find(text)
    return match?.groupValues?.get(1) ?: "0.1.0"
}

fun hermesVersionCode(): Int {
    if (releaseTag.isBlank()) {
        return 1
    }
    val releaseMatch = Regex("""v(\d{4})\.(\d{1,2})\.(\d{1,2})(?:\.(\d{1,2}))?""").matchEntire(releaseTag)
        ?: return 1
    val year = releaseMatch.groupValues[1]
    val month = releaseMatch.groupValues[2].padStart(2, '0')
    val day = releaseMatch.groupValues[3].padStart(2, '0')
    val seq = releaseMatch.groupValues[4].ifBlank { "0" }.padStart(2, '0')
    return "$year$month$day$seq".toInt()
}

android {
    namespace = "com.nousresearch.hermesagent"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.nousresearch.hermesagent"
        minSdk = 24
        targetSdk = 35
        versionCode = hermesVersionCode()
        versionName = hermesVersionName()
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        vectorDrawables {
            useSupportLibrary = true
        }
        ndk {
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
    }

    signingConfigs {
        if (hasReleaseKeystore) {
            create("release") {
                storeFile = rootDir.resolve(keystoreProperties.getProperty("storeFile"))
                storePassword = keystoreProperties.getProperty("storePassword")
                keyAlias = keystoreProperties.getProperty("keyAlias")
                keyPassword = keystoreProperties.getProperty("keyPassword")
            }
        }
    }

    buildTypes {
        release {
            signingConfig = if (hasReleaseKeystore) {
                signingConfigs.getByName("release")
            } else {
                signingConfigs.getByName("debug")
            }
            isMinifyEnabled = false
            isShrinkResources = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        compose = true
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
}

chaquopy {
    defaultConfig {
        version = "3.11"

        val configuredBuildPython = System.getenv("PYTHON_FOR_BUILD")
        if (!configuredBuildPython.isNullOrBlank()) {
            buildPython(configuredBuildPython)
        } else {
            buildPython("python3.11")
        }

        pip {
            install("../../[android]")
        }
    }
}

dependencies {
    val composeBom = platform("androidx.compose:compose-bom:2024.12.01")

    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
    implementation("androidx.lifecycle:lifecycle-viewmodel-ktx:2.8.7")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.7")
    implementation("androidx.activity:activity-compose:1.9.3")
    implementation(composeBom)
    androidTestImplementation(composeBom)
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3")
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation("com.squareup.okhttp3:okhttp-sse:4.12.0")
    implementation("androidx.security:security-crypto:1.1.0-alpha06")

    testImplementation("junit:junit:4.13.2")
    testImplementation("com.squareup.okhttp3:mockwebserver:4.12.0")
    androidTestImplementation("androidx.test:core-ktx:1.6.1")
    androidTestImplementation("androidx.test.ext:junit:1.3.0")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.7.0")
    androidTestImplementation("androidx.compose.ui:ui-test-junit4")
    debugImplementation("androidx.compose.ui:ui-tooling")
    debugImplementation("androidx.compose.ui:ui-test-manifest")
}
