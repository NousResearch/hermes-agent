plugins { id("com.android.application") }

android {
    namespace = "com.nousresearch.hermes.notebook"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.nousresearch.hermes.notebook"
        minSdk = 26
        targetSdk = 36
        versionCode = 1
        versionName = "0.1.0"
        testInstrumentationRunner = "android.test.InstrumentationTestRunner"
    }

    buildFeatures { buildConfig = true }
    buildTypes {
        release {
            isMinifyEnabled = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

dependencies {
    implementation("com.google.mlkit:digital-ink-recognition:19.0.0")
    testImplementation("junit:junit:4.13.2")
}
