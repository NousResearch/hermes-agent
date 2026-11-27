# Activity Diagram Guide

Activity diagrams show workflows, business processes, and algorithmic flows. They're perfect for documenting user actions, system processes, and decision trees.

## When to Use Activity Diagrams

- **Business workflows**: Order processing, approval flows, customer journeys
- **System processes**: Data pipelines, ETL jobs, batch processing
- **User interactions**: Login flows, checkout processes, form submissions
- **Decision trees**: Conditional logic, branching scenarios
- **Algorithm visualization**: Step-by-step processes with conditions

## Basic Syntax

### Simple Linear Flow

```mermaid
flowchart TD
    Start([Start Process]) --> Step1[Validate Input]
    Step1 --> Step2[Process Data]
    Step2 --> Step3[Save Results]
    Step3 --> End([End Process])

    classDef startEnd fill:#E6E6FA,stroke:#333,stroke-width:2px,color:darkblue
    classDef process fill:#90EE90,stroke:#333,stroke-width:2px,color:darkgreen

    class Start,End startEnd
    class Step1,Step2,Step3 process
```

### Decision Points

```mermaid
flowchart TD
    Start([Start]) --> Input[Get User Input]
    Input --> Valid{Input Valid?}
    Valid -->|Yes| Process[Process Request]
    Valid -->|No| Error[Show Error]
    Process --> Save{Save Success?}
    Save -->|Yes| Success[Show Success Message]
    Save -->|No| Retry{Retry?}
    Retry -->|Yes| Process
    Retry -->|No| Fail[Show Failure]
    Error --> End([End])
    Success --> End
    Fail --> End

    classDef startEnd fill:#E6E6FA,stroke:#333,stroke-width:2px,color:darkblue
    classDef process fill:#90EE90,stroke:#333,stroke-width:2px,color:darkgreen
    classDef decision fill:#FFD700,stroke:#333,stroke-width:2px,color:black
    classDef error fill:#FFB6C1,stroke:#DC143C,stroke-width:2px,color:black
    classDef success fill:#90EE90,stroke:#2E7D2E,stroke-width:2px,color:darkgreen

    class Start,End startEnd
    class Input,Process process
    class Valid,Save,Retry decision
    class Error,Fail error
    class Success success
```

## Common Patterns

### Authentication Flow

```mermaid
flowchart TD
    Start([User Visits Site]) --> CheckAuth{Authenticated?}
    CheckAuth -->|Yes| Dashboard[Show Dashboard]
    CheckAuth -->|No| Login[Show Login Page]
    Login --> Submit[User Submits Credentials]
    Submit --> Validate{Valid Credentials?}
    Validate -->|Yes| CreateSession[Create Session]
    Validate -->|No| LoginError[Show Error Message]
    LoginError --> Login
    CreateSession --> SetCookie[Set Auth Cookie]
    SetCookie --> Dashboard
    Dashboard --> End([User Interacts])

    classDef startEnd fill:#E6E6FA,stroke:#333,stroke-width:2px,color:darkblue
    classDef process fill:#90EE90,stroke:#333,stroke-width:2px,color:darkgreen
    classDef decision fill:#FFD700,stroke:#333,stroke-width:2px,color:black
    classDef error fill:#FFB6C1,stroke:#DC143C,stroke-width:2px,color:black

    class Start,End startEnd
    class Login,Submit,CreateSession,SetCookie,Dashboard process
    class CheckAuth,Validate decision
    class LoginError error
```

### E-commerce Checkout Flow

```mermaid
flowchart TD
    Start([User Adds Items]) --> Cart[View Cart]
    Cart --> Checkout[Click Checkout]
    Checkout --> Login{Logged In?}
    Login -->|No| Auth[Login/Register]
    Login -->|Yes| Shipping
    Auth --> Shipping[Enter Shipping Info]
    Shipping --> Payment[Enter Payment Info]
    Payment --> Review[Review Order]
    Review --> Confirm{Confirm?}
    Confirm -->|No| Cart
    Confirm -->|Yes| Process[Process Payment]
    Process --> Result{Payment Success?}
    Result -->|Yes| CreateOrder[Create Order]
    Result -->|No| PayError[Show Payment Error]
    PayError --> Payment
    CreateOrder --> SendEmail[Send Confirmation Email]
    SendEmail --> Success([Order Complete])

    classDef startEnd fill:#E6E6FA,stroke:#333,stroke-width:2px,color:darkblue
    classDef process fill:#90EE90,stroke:#333,stroke-width:2px,color:darkgreen
    classDef decision fill:#FFD700,stroke:#333,stroke-width:2px,color:black
    classDef error fill:#FFB6C1,stroke:#DC143C,stroke-width:2px,color:black

    class Start,Success startEnd
    class Cart,Checkout,Auth,Shipping,Payment,Review,Process,CreateOrder,SendEmail process
    class Login,Confirm,Result decision
    class PayError error
```

### Data Processing Pipeline

```mermaid
flowchart TD
    Start([Start ETL Job]) --> Extract[Extract Data from Source]
    Extract --> ValidateSchema{Schema Valid?}
    ValidateSchema -->|No| SchemaError[Log Schema Error]
    ValidateSchema -->|Yes| Transform[Transform Data]
    SchemaError --> Alert[Send Alert]
    Transform --> ValidateData{Data Quality OK?}
    ValidateData -->|No| DataError[Log Data Quality Issues]
    ValidateData -->|Yes| Load[Load to Data Warehouse]
    DataError --> Quarantine[Move to Quarantine]
    Load --> LoadResult{Load Success?}
    LoadResult -->|No| LoadError[Handle Load Error]
    LoadResult -->|Yes| UpdateMetrics[Update Job Metrics]
    LoadError --> Retry{Retry?}
    Retry -->|Yes| Load
    Retry -->|No| Fail[Mark Job Failed]
    UpdateMetrics --> Success([Job Complete])
    Quarantine --> End([End with Errors])
    Alert --> End
    Fail --> End

    classDef startEnd fill:#E6E6FA,stroke:#333,stroke-width:2px,color:darkblue
    classDef process fill:#90EE90,stroke:#333,stroke-width:2px,color:darkgreen
    classDef decision fill:#FFD700,stroke:#333,stroke-width:2px,color:black
    classDef error fill:#FFB6C1,stroke:#DC143C,stroke-width:2px,color:black

    class Start,Success,End startEnd
    class Extract,Transform,Load,UpdateMetrics process
    class ValidateSchema,ValidateData,LoadResult,Retry decision
    class SchemaError,DataError,LoadError,Quarantine,Alert,Fail error
```

## Unicode Semantic Symbols

Enhance activity diagrams with Unicode symbols for visual clarity:

### Process Types

```mermaid
flowchart TD
    Start([⚡ Start Process]) --> Input[📥 Receive Input]
    Input --> Validate[✓ Validate Data]
    Validate --> Process[⚙️ Process Request]
    Process --> Save[💾 Save to Database]
    Save --> Notify[📧 Send Notification]
    Notify --> End([✓ Complete])

    classDef startEnd fill:#E6E6FA,stroke:#333,stroke-width:2px,color:darkblue
    classDef process fill:#90EE90,stroke:#333,stroke-width:2px,color:darkgreen

    class Start,End startEnd
    class Input,Validate,Process,Save,Notify process
```

### Error Handling with Symbols

```mermaid
flowchart TD
    Start([🚀 Start]) --> API[🌐 Call External API]
    API --> Result{📊 Status?}
    Result -->|✓ Success| Parse[📝 Parse Response]
    Result -->|❌ Error| RetryCheck{🔄 Retry?}
    RetryCheck -->|Yes| Wait[⏱️ Wait]
    RetryCheck -->|No| Alert[🚨 Alert Admin]
    Wait --> API
    Parse --> Store[💾 Store Data]
    Store --> Complete([✓ Done])
    Alert --> Fail([❌ Failed])

    classDef startEnd fill:#E6E6FA,stroke:#333,stroke-width:2px,color:darkblue
    classDef process fill:#90EE90,stroke:#333,stroke-width:2px,color:darkgreen
    classDef decision fill:#FFD700,stroke:#333,stroke-width:2px,color:black
    classDef error fill:#FFB6C1,stroke:#DC143C,stroke-width:2px,color:black

    class Start,Complete startEnd
    class API,Parse,Store,Wait process
    class Result,RetryCheck decision
    class Alert,Fail error
```

## Common Unicode Symbols for Activities

| Symbol | Meaning | Use Case |
|--------|---------|----------|
| ⚡ | Start/Trigger | Process initiation |
| ✓ / ✅ | Success | Successful completion |
| ❌ / ✗ | Failure | Error states |
| ⚙️ | Processing | Active computation |
| 💾 | Storage | Database/file operations |
| 📥 | Input | Receiving data |
| 📤 | Output | Sending data |
| 🌐 | Network | API calls, web requests |
| 📧 | Email | Notification/messaging |
| 🔄 | Retry | Retry logic |
| ⏱️ | Wait | Delays, timeouts |
| 🚨 | Alert | Critical notifications |
| 📊 | Analysis | Data processing |
| 🔐 | Security | Authentication/encryption |
| 📝 | Logging | Write logs |
| 🎯 | Target | Goal achievement |

## Swimlane Activity Diagrams

Show responsibilities across different actors or systems:

```mermaid
flowchart TD
    subgraph User
        U1[Browse Products]
        U2[Add to Cart]
        U3[Checkout]
        U4[Enter Payment]
    end

    subgraph Frontend
        F1[Display Catalog]
        F2[Update Cart UI]
        F3[Show Checkout Form]
        F4[Submit Order]
    end

    subgraph Backend
        B1[Fetch Products]
        B2[Update Cart Session]
        B3[Validate Order]
        B4[Process Payment]
        B5[Create Order]
    end

    subgraph Database
        D1[Query Products]
        D2[Save Cart]
        D3[Insert Order]
    end

    U1 --> F1
    F1 --> B1
    B1 --> D1
    D1 --> B1
    B1 --> F1
    F1 --> U1

    U2 --> F2
    F2 --> B2
    B2 --> D2

    U3 --> F3
    F3 --> B3

    U4 --> F4
    F4 --> B4
    B4 --> B5
    B5 --> D3

    classDef user fill:#FFE4B5,stroke:#333,stroke-width:2px,color:black
    classDef frontend fill:#87CEEB,stroke:#333,stroke-width:2px,color:darkblue
    classDef backend fill:#90EE90,stroke:#333,stroke-width:2px,color:darkgreen
    classDef database fill:#E6E6FA,stroke:#333,stroke-width:2px,color:darkblue

    class U1,U2,U3,U4 user
    class F1,F2,F3,F4 frontend
    class B1,B2,B3,B4,B5 backend
    class D1,D2,D3 database
```

## Best Practices

### 1. Keep Flows Focused

- **Single purpose**: One activity diagram = one workflow
- **Limit complexity**: Max 15-20 nodes per diagram
- **Break down**: Split complex flows into multiple diagrams

### 2. Use Clear Naming

```mermaid
flowchart TD
    %% Good: Clear, action-oriented labels
    Start([Start Order Process])
    Validate[Validate Order Items]
    Calculate[Calculate Total Price]

    %% Avoid: Vague labels
    %% Process1[Process]
    %% DoStuff[Do Stuff]
```

### 3. Consistent Decision Logic

```mermaid
flowchart TD
    Check{Is Valid?}
    Check -->|Yes| Success[Continue]
    Check -->|No| Error[Show Error]

    %% Always use Yes/No, True/False, or Success/Failure
    %% Be consistent throughout the diagram
```

### 4. Handle Error Paths

Always show what happens when things go wrong:

```mermaid
flowchart TD
    Start([Start]) --> Process[Process Request]
    Process --> Result{Success?}
    Result -->|Yes| Happy[✓ Complete]
    Result -->|No| Retry{Can Retry?}
    Retry -->|Yes| Process
    Retry -->|No| Fail[❌ Mark Failed]
    Fail --> Alert[🚨 Alert Admin]
    Alert --> End([End])
    Happy --> End
```

### 5. Use Subgraphs for Organization

```mermaid
flowchart TD
    Start([Start])

    subgraph Validation
        V1[Check Input]
        V2[Validate Schema]
        V3[Check Permissions]
    end

    subgraph Processing
        P1[Transform Data]
        P2[Enrich Data]
        P3[Calculate Results]
    end

    subgraph Persistence
        S1[Save to DB]
        S2[Update Cache]
        S3[Index Search]
    end

    Start --> V1
    V1 --> V2
    V2 --> V3
    V3 --> P1
    P1 --> P2
    P2 --> P3
    P3 --> S1
    S1 --> S2
    S2 --> S3
    S3 --> End([End])
```

## Activity Diagram Templates

### Template: API Request Flow

```mermaid
flowchart TD
    Start([API Request Received]) --> Auth{Authenticated?}
    Auth -->|No| Unauthorized[401 Unauthorized]
    Auth -->|Yes| Validate{Valid Input?}
    Validate -->|No| BadRequest[400 Bad Request]
    Validate -->|Yes| Process[Process Request]
    Process --> DBCall{DB Available?}
    DBCall -->|No| ServerError[503 Service Unavailable]
    DBCall -->|Yes| Execute[Execute Query]
    Execute --> Format[Format Response]
    Format --> Success[200 OK]
    Success --> End([Return Response])
    Unauthorized --> End
    BadRequest --> End
    ServerError --> End
```

### Template: Batch Job Execution

```mermaid
flowchart TD
    Start([Scheduled Start]) --> Lock{Acquire Lock?}
    Lock -->|No| Skip[Skip - Already Running]
    Lock -->|Yes| LoadConfig[Load Configuration]
    LoadConfig --> Validate{Config Valid?}
    Validate -->|No| ConfigError[Log Config Error]
    Validate -->|Yes| FetchData[Fetch Source Data]
    FetchData --> DataCheck{Data Available?}
    DataCheck -->|No| NoData[Log No Data]
    DataCheck -->|Yes| Transform[Transform Data]
    Transform --> Process[Process Records]
    Process --> Save{Save Success?}
    Save -->|No| SaveError[Handle Save Error]
    Save -->|Yes| Metrics[Update Metrics]
    SaveError --> Retry{Should Retry?}
    Retry -->|Yes| Process
    Retry -->|No| Fail[Mark Job Failed]
    Metrics --> ReleaseLock[Release Lock]
    ReleaseLock --> Success([Job Complete])
    ConfigError --> End([Job Skipped])
    NoData --> End
    Skip --> End
    Fail --> End
```

## Anti-Patterns to Avoid

### ❌ Too Complex

```mermaid
flowchart TD
    %% Avoid: Too many nodes, hard to follow
    A --> B --> C --> D --> E --> F --> G --> H --> I --> J
    J --> K --> L --> M --> N --> O --> P --> Q --> R --> S
```

**Fix**: Break into multiple focused diagrams

### ❌ Unclear Decisions

```mermaid
flowchart TD
    Check{Check Something}
    Check --> A[Action A]
    Check --> B[Action B]
    %% Missing: What condition leads to A vs B?
```

**Fix**: Label all decision branches clearly

### ❌ Missing Error Handling

```mermaid
flowchart TD
    Start([Start]) --> Process[Process]
    Process --> End([End])
    %% Missing: What if Process fails?
```

**Fix**: Always include error paths

## Integration with Code

Activity diagrams should reflect actual code workflows. See the `code-to-diagram` guide for examples of generating activity diagrams from:

- Python functions with control flow
- Java/Spring Boot request handlers
- Node.js/Express route handlers
- React component lifecycle

---

**Next Steps:**
- See `deployment-diagrams.md` for infrastructure flows
- See `code-to-diagram/` for code-to-diagram examples
- See `unicode-symbols/guide.md` for complete symbol reference
