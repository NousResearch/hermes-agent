# Deployment Diagram Guide

Deployment diagrams visualize infrastructure, server architecture, network topology, and how software components are deployed across physical or cloud resources.

## When to Use Deployment Diagrams

- **Infrastructure architecture**: Cloud resources, server layouts, network topology
- **Deployment configurations**: How services are distributed across environments
- **System boundaries**: External dependencies, third-party integrations
- **Scaling strategies**: Load balancing, replication, failover
- **Network flows**: Data paths, communication channels, security zones

## Basic Syntax

### Simple Deployment

```mermaid
graph TB
    subgraph "Production Environment"
        LB[Load Balancer<br/>📊 NGINX]
        subgraph "Application Tier"
            App1[App Server 1<br/>⚙️ Node.js]
            App2[App Server 2<br/>⚙️ Node.js]
        end
        subgraph "Data Tier"
            DB[(Database<br/>💾 PostgreSQL)]
            Cache[(Cache<br/>⚡ Redis)]
        end
    end

    Client[👤 Client] --> LB
    LB --> App1
    LB --> App2
    App1 --> DB
    App1 --> Cache
    App2 --> DB
    App2 --> Cache

    classDef client fill:#FFE4B5,stroke:#333,stroke-width:2px,color:black
    classDef loadbalancer fill:#87CEEB,stroke:#333,stroke-width:2px,color:darkblue
    classDef appServer fill:#90EE90,stroke:#333,stroke-width:2px,color:darkgreen
    classDef database fill:#E6E6FA,stroke:#333,stroke-width:2px,color:darkblue

    class Client client
    class LB loadbalancer
    class App1,App2 appServer
    class DB,Cache database
```

## Common Deployment Patterns

### Three-Tier Architecture

```mermaid
graph TB
    subgraph "Cloud Provider: AWS"
        subgraph "Public Subnet"
            ALB[🌐 Application Load Balancer<br/>Port 443]
        end

        subgraph "Private Subnet - App Tier"
            EC2_1[⚙️ EC2 Instance 1<br/>us-east-1a<br/>t3.large]
            EC2_2[⚙️ EC2 Instance 2<br/>us-east-1b<br/>t3.large]
            ASG[📊 Auto Scaling Group<br/>Min: 2, Max: 10]
        end

        subgraph "Private Subnet - Data Tier"
            RDS[(💾 RDS PostgreSQL<br/>Primary<br/>db.r5.xlarge)]
            RDS_R[(💾 RDS Read Replica<br/>us-east-1b)]
            ElastiCache[(⚡ ElastiCache Redis<br/>cache.r5.large)]
        end

        subgraph "Storage"
            S3[📦 S3 Bucket<br/>Static Assets]
        end
    end

    Internet[🌍 Internet] --> ALB
    ALB --> EC2_1
    ALB --> EC2_2
    EC2_1 --> RDS
    EC2_2 --> RDS
    EC2_1 --> RDS_R
    EC2_2 --> RDS_R
    EC2_1 --> ElastiCache
    EC2_2 --> ElastiCache
    EC2_1 --> S3
    EC2_2 --> S3
    RDS --> RDS_R

    ASG -.manages.-> EC2_1
    ASG -.manages.-> EC2_2

    classDef public fill:#FFE4B5,stroke:#333,stroke-width:2px,color:black
    classDef compute fill:#90EE90,stroke:#333,stroke-width:2px,color:darkgreen
    classDef database fill:#E6E6FA,stroke:#333,stroke-width:2px,color:darkblue
    classDef storage fill:#FFD700,stroke:#333,stroke-width:2px,color:black

    class Internet,ALB public
    class EC2_1,EC2_2,ASG compute
    class RDS,RDS_R,ElastiCache database
    class S3 storage
```

### Microservices Deployment (Kubernetes)

```mermaid
graph TB
    subgraph "Kubernetes Cluster"
        subgraph "Ingress"
            Ingress[🌐 NGINX Ingress<br/>ingress-nginx]
        end

        subgraph "Namespace: production"
            subgraph "Auth Service"
                AuthPod1[⚙️ auth-service-1]
                AuthPod2[⚙️ auth-service-2]
                AuthSvc[Service: auth-svc<br/>ClusterIP]
            end

            subgraph "API Service"
                APIPod1[⚙️ api-service-1]
                APIPod2[⚙️ api-service-2]
                APIPod3[⚙️ api-service-3]
                APISvc[Service: api-svc<br/>ClusterIP]
            end

            subgraph "Worker Service"
                WorkerPod1[⚙️ worker-1]
                WorkerPod2[⚙️ worker-2]
            end
        end

        subgraph "External Dependencies"
            RabbitMQ[🐰 RabbitMQ<br/>message-queue]
            PostgreSQL[(💾 PostgreSQL<br/>Cloud SQL)]
            Redis[(⚡ Redis<br/>Memorystore)]
        end
    end

    Client[👤 Client] --> Ingress
    Ingress --> AuthSvc
    Ingress --> APISvc
    AuthSvc --> AuthPod1
    AuthSvc --> AuthPod2
    APISvc --> APIPod1
    APISvc --> APIPod2
    APISvc --> APIPod3

    AuthPod1 --> PostgreSQL
    AuthPod2 --> PostgreSQL
    APIPod1 --> PostgreSQL
    APIPod2 --> PostgreSQL
    APIPod3 --> PostgreSQL

    APIPod1 --> Redis
    APIPod2 --> Redis
    APIPod3 --> Redis

    APIPod1 --> RabbitMQ
    APIPod2 --> RabbitMQ
    APIPod3 --> RabbitMQ

    RabbitMQ --> WorkerPod1
    RabbitMQ --> WorkerPod2
    WorkerPod1 --> PostgreSQL
    WorkerPod2 --> PostgreSQL

    classDef client fill:#FFE4B5,stroke:#333,stroke-width:2px,color:black
    classDef ingress fill:#87CEEB,stroke:#333,stroke-width:2px,color:darkblue
    classDef service fill:#90EE90,stroke:#333,stroke-width:2px,color:darkgreen
    classDef database fill:#E6E6FA,stroke:#333,stroke-width:2px,color:darkblue
    classDef queue fill:#FFD700,stroke:#333,stroke-width:2px,color:black

    class Client client
    class Ingress ingress
    class AuthPod1,AuthPod2,AuthSvc,APIPod1,APIPod2,APIPod3,APISvc,WorkerPod1,WorkerPod2 service
    class PostgreSQL,Redis database
    class RabbitMQ queue
```

### Serverless Architecture

```mermaid
graph TB
    subgraph "AWS Serverless Architecture"
        subgraph "Edge"
            CloudFront[☁️ CloudFront<br/>CDN Distribution]
            S3Web[📦 S3 Static Hosting<br/>React Frontend]
        end

        subgraph "API Layer"
            APIGW[🌐 API Gateway<br/>REST API]
        end

        subgraph "Compute"
            Lambda1[⚡ Lambda: Auth<br/>Node.js 18]
            Lambda2[⚡ Lambda: Users<br/>Node.js 18]
            Lambda3[⚡ Lambda: Orders<br/>Python 3.11]
            Lambda4[⚡ Lambda: Processor<br/>Java 17]
        end

        subgraph "Data"
            DynamoDB[(💾 DynamoDB<br/>Orders Table)]
            RDS[(💾 RDS Aurora<br/>Users DB)]
            S3Data[📦 S3 Bucket<br/>Order Files]
        end

        subgraph "Events"
            EventBridge[📨 EventBridge<br/>Event Bus]
            SQS[📬 SQS Queue<br/>Order Processing]
        end
    end

    Client[👤 Client] --> CloudFront
    CloudFront --> S3Web
    CloudFront --> APIGW

    APIGW --> Lambda1
    APIGW --> Lambda2
    APIGW --> Lambda3

    Lambda1 --> RDS
    Lambda2 --> RDS
    Lambda3 --> DynamoDB
    Lambda3 --> EventBridge

    EventBridge --> SQS
    SQS --> Lambda4
    Lambda4 --> S3Data
    Lambda4 --> DynamoDB

    classDef edge fill:#87CEEB,stroke:#333,stroke-width:2px,color:darkblue
    classDef compute fill:#90EE90,stroke:#333,stroke-width:2px,color:darkgreen
    classDef database fill:#E6E6FA,stroke:#333,stroke-width:2px,color:darkblue
    classDef events fill:#FFD700,stroke:#333,stroke-width:2px,color:black
    classDef client fill:#FFE4B5,stroke:#333,stroke-width:2px,color:black

    class Client client
    class CloudFront,S3Web,APIGW edge
    class Lambda1,Lambda2,Lambda3,Lambda4 compute
    class DynamoDB,RDS,S3Data database
    class EventBridge,SQS events
```

## Infrastructure as Code Mapping

### Pulumi/Terraform to Deployment Diagram

When you have IaC code, map resources to deployment diagrams:

```mermaid
graph TB
    subgraph "GCP Project: my-project"
        subgraph "VPC: main-vpc"
            subgraph "Subnet: private-subnet"
                CloudRun1[☁️ Cloud Run Service<br/>contacts-api<br/>CPU: 1, Mem: 512Mi]
                CloudRun2[☁️ Cloud Run Service<br/>auth-api<br/>CPU: 2, Mem: 1Gi]
            end
        end

        subgraph "VPC Peering"
            VPCConnector[🔌 Serverless VPC Connector<br/>min: 2, max: 10]
        end

        subgraph "Database Cluster"
            AlloyDB[(💾 AlloyDB Cluster<br/>Primary Instance<br/>8 vCPU, 32GB)]
            AlloyDBRead[(💾 AlloyDB Read Pool<br/>4 vCPU, 16GB<br/>Replicas: 2)]
        end

        subgraph "Security"
            IAM[🔐 Service Account<br/>cloud-run-sa<br/>Roles: AlloyDB Client]
            SecretManager[🔑 Secret Manager<br/>DB Credentials]
        end

        subgraph "Monitoring"
            CloudLogging[📝 Cloud Logging]
            CloudMonitoring[📊 Cloud Monitoring]
        end
    end

    Internet[🌍 Internet] --> CloudRun1
    Internet --> CloudRun2

    CloudRun1 --> VPCConnector
    CloudRun2 --> VPCConnector
    VPCConnector --> AlloyDB
    VPCConnector --> AlloyDBRead

    CloudRun1 --> SecretManager
    CloudRun2 --> SecretManager

    IAM -.identity.-> CloudRun1
    IAM -.identity.-> CloudRun2

    CloudRun1 --> CloudLogging
    CloudRun2 --> CloudLogging
    AlloyDB --> CloudMonitoring
    AlloyDBRead --> CloudMonitoring

    classDef public fill:#FFE4B5,stroke:#333,stroke-width:2px,color:black
    classDef compute fill:#90EE90,stroke:#333,stroke-width:2px,color:darkgreen
    classDef database fill:#E6E6FA,stroke:#333,stroke-width:2px,color:darkblue
    classDef security fill:#FFB6C1,stroke:#DC143C,stroke-width:2px,color:black
    classDef monitoring fill:#F0E68C,stroke:#333,stroke-width:2px,color:black

    class Internet public
    class CloudRun1,CloudRun2,VPCConnector compute
    class AlloyDB,AlloyDBRead database
    class IAM,SecretManager security
    class CloudLogging,CloudMonitoring monitoring
```

### Docker Compose to Deployment Diagram

```mermaid
graph TB
    subgraph "Docker Compose Stack"
        subgraph "Frontend Network"
            Nginx[🌐 nginx:alpine<br/>Port: 80:80<br/>reverse-proxy]
            React[⚛️ react-app:latest<br/>Port: 3000<br/>volumes: ./app]
        end

        subgraph "Backend Network"
            API[⚙️ api:latest<br/>Port: 8000<br/>env: production]
            Worker[⚙️ worker:latest<br/>replicas: 3]
        end

        subgraph "Data Network"
            Postgres[(💾 postgres:15<br/>Port: 5432<br/>volumes: pgdata)]
            Redis[(⚡ redis:7-alpine<br/>Port: 6379<br/>maxmemory: 256mb)]
            RabbitMQ[🐰 rabbitmq:3-management<br/>Port: 5672, 15672]
        end

        subgraph "Monitoring"
            Prometheus[📊 prom/prometheus<br/>Port: 9090]
            Grafana[📈 grafana/grafana<br/>Port: 3001]
        end
    end

    Client[👤 Client] --> Nginx
    Nginx --> React
    Nginx --> API

    API --> Postgres
    API --> Redis
    API --> RabbitMQ

    RabbitMQ --> Worker
    Worker --> Postgres

    API --> Prometheus
    Worker --> Prometheus
    Prometheus --> Grafana

    classDef client fill:#FFE4B5,stroke:#333,stroke-width:2px,color:black
    classDef frontend fill:#87CEEB,stroke:#333,stroke-width:2px,color:darkblue
    classDef backend fill:#90EE90,stroke:#333,stroke-width:2px,color:darkgreen
    classDef database fill:#E6E6FA,stroke:#333,stroke-width:2px,color:darkblue
    classDef monitoring fill:#F0E68C,stroke:#333,stroke-width:2px,color:black

    class Client client
    class Nginx,React frontend
    class API,Worker backend
    class Postgres,Redis,RabbitMQ database
    class Prometheus,Grafana monitoring
```

## Network Security Zones

```mermaid
graph TB
    subgraph "Public Zone - DMZ"
        WAF[🛡️ WAF / DDoS Protection<br/>CloudFlare]
        LB[🌐 Load Balancer<br/>Public IP]
    end

    subgraph "Application Zone - Private"
        WebServer1[⚙️ Web Server 1<br/>10.0.1.10]
        WebServer2[⚙️ Web Server 2<br/>10.0.1.11]
        AppServer1[⚙️ App Server 1<br/>10.0.2.10]
        AppServer2[⚙️ App Server 2<br/>10.0.2.11]
    end

    subgraph "Database Zone - Restricted"
        DBPrimary[(💾 DB Primary<br/>10.0.3.10)]
        DBReplica[(💾 DB Replica<br/>10.0.3.11)]
    end

    subgraph "Management Zone"
        Bastion[🔧 Bastion Host<br/>SSH Gateway<br/>10.0.4.10]
        Monitoring[📊 Monitoring<br/>10.0.4.20]
    end

    Internet[🌍 Internet] -->|HTTPS 443| WAF
    WAF -->|filtered| LB
    LB -->|HTTP 8080| WebServer1
    LB -->|HTTP 8080| WebServer2

    WebServer1 -->|HTTP 9000| AppServer1
    WebServer2 -->|HTTP 9000| AppServer2

    AppServer1 -->|PostgreSQL 5432| DBPrimary
    AppServer2 -->|PostgreSQL 5432| DBPrimary
    DBPrimary -->|replication| DBReplica

    Admin[👨‍💼 Admin] -->|SSH 22| Bastion
    Bastion -.SSH.-> WebServer1
    Bastion -.SSH.-> WebServer2
    Bastion -.SSH.-> AppServer1
    Bastion -.SSH.-> AppServer2

    WebServer1 -.metrics.-> Monitoring
    WebServer2 -.metrics.-> Monitoring
    AppServer1 -.metrics.-> Monitoring
    AppServer2 -.metrics.-> Monitoring
    DBPrimary -.metrics.-> Monitoring
    DBReplica -.metrics.-> Monitoring

    classDef public fill:#FFE4B5,stroke:#333,stroke-width:2px,color:black
    classDef app fill:#90EE90,stroke:#333,stroke-width:2px,color:darkgreen
    classDef database fill:#E6E6FA,stroke:#333,stroke-width:2px,color:darkblue
    classDef management fill:#F0E68C,stroke:#333,stroke-width:2px,color:black

    class Internet,WAF,LB public
    class WebServer1,WebServer2,AppServer1,AppServer2 app
    class DBPrimary,DBReplica database
    class Bastion,Monitoring,Admin management
```

## Unicode Symbols for Infrastructure

| Symbol | Meaning | Use Case |
|--------|---------|----------|
| ☁️ | Cloud Service | Cloud resources, SaaS |
| 🌐 | Load Balancer | LB, API Gateway |
| ⚙️ | Application Server | Compute instances |
| 💾 | Database | Persistent storage |
| ⚡ | Cache | Redis, Memcached |
| 📦 | Object Storage | S3, GCS, Blob Storage |
| 🐰 | Message Queue | RabbitMQ, SQS |
| 🔌 | Network Connector | VPN, VPC peering |
| 🛡️ | Security | Firewall, WAF |
| 🔐 | IAM/Auth | Identity, secrets |
| 🔑 | Secrets | API keys, passwords |
| 📊 | Monitoring | Metrics, dashboards |
| 📝 | Logging | Log aggregation |
| 🔄 | Replication | DB replication, sync |
| 🌍 | Internet | Public access |
| 👤 | User/Client | End users |
| 🔧 | Management | Admin tools, bastion |
| ⚛️ | Frontend | React, Vue, Angular |
| 📨 | Event Bus | EventBridge, Pub/Sub |
| 📬 | Queue | Job queues |

## Best Practices

### 1. Show Resource Specifications

```mermaid
graph TB
    EC2[⚙️ EC2 Instance<br/>Type: t3.xlarge<br/>CPU: 4 vCPU<br/>Memory: 16GB<br/>AZ: us-east-1a]

    RDS[(💾 RDS PostgreSQL<br/>Instance: db.r5.2xlarge<br/>vCPU: 8<br/>RAM: 64GB<br/>Storage: 1TB SSD<br/>Multi-AZ: Yes)]
```

### 2. Indicate Network Boundaries

Use subgraphs for:
- VPCs / Virtual Networks
- Subnets (public/private)
- Security groups
- Availability zones
- Regions

### 3. Show Scaling Configuration

```mermaid
graph TB
    ASG[📊 Auto Scaling Group<br/>Min: 2<br/>Desired: 4<br/>Max: 20<br/>Target CPU: 70%]

    ASG -.manages.-> Instance1[⚙️ Instance 1]
    ASG -.manages.-> Instance2[⚙️ Instance 2]
    ASG -.manages.-> Instance3[⚙️ Instance 3]
    ASG -.manages.-> Instance4[⚙️ Instance 4]
```

### 4. Document Ports and Protocols

```mermaid
graph LR
    Client -->|HTTPS 443| ALB
    ALB -->|HTTP 8080| App
    App -->|PostgreSQL 5432| DB
    App -->|Redis 6379| Cache
```

### 5. Indicate High Availability

```mermaid
graph TB
    subgraph "Region: us-east-1"
        subgraph "AZ: us-east-1a"
            App1[⚙️ App Server 1]
            DB1[(💾 DB Primary)]
        end

        subgraph "AZ: us-east-1b"
            App2[⚙️ App Server 2]
            DB2[(💾 DB Standby)]
        end

        subgraph "AZ: us-east-1c"
            App3[⚙️ App Server 3]
        end
    end

    LB[🌐 Load Balancer] --> App1
    LB --> App2
    LB --> App3

    App1 --> DB1
    App2 --> DB1
    App3 --> DB1

    DB1 -.replication.-> DB2
```

## Deployment Diagram Templates

### Template: Multi-Region Deployment

```mermaid
graph TB
    subgraph "Global"
        Route53[🌐 Route 53<br/>DNS / Routing<br/>Latency-based]
        CloudFront[☁️ CloudFront<br/>Global CDN]
    end

    subgraph "Region: US-East"
        ALB_US[🌐 ALB US]
        App_US1[⚙️ App US-1]
        App_US2[⚙️ App US-2]
        RDS_US[(💾 RDS US<br/>Primary)]
    end

    subgraph "Region: EU-West"
        ALB_EU[🌐 ALB EU]
        App_EU1[⚙️ App EU-1]
        App_EU2[⚙️ App EU-2]
        RDS_EU[(💾 RDS EU<br/>Replica)]
    end

    Client[👤 Client] --> Route53
    Route53 --> CloudFront
    CloudFront --> ALB_US
    CloudFront --> ALB_EU

    ALB_US --> App_US1
    ALB_US --> App_US2
    ALB_EU --> App_EU1
    ALB_EU --> App_EU2

    App_US1 --> RDS_US
    App_US2 --> RDS_US
    App_EU1 --> RDS_EU
    App_EU2 --> RDS_EU

    RDS_US -.cross-region replication.-> RDS_EU
```

### Template: Hybrid Cloud

```mermaid
graph TB
    subgraph "On-Premises Data Center"
        OnPremApp[⚙️ Legacy Application<br/>Windows Server]
        OnPremDB[(💾 Oracle Database<br/>RAC Cluster)]
        VPN[🔌 VPN Gateway]
    end

    subgraph "AWS Cloud"
        subgraph "VPC"
            VPNGateway[🔌 AWS VPN Gateway]
            CloudApp[⚙️ Cloud Application<br/>ECS Fargate]
            RDS[(💾 RDS MySQL)]
        end

        API[🌐 API Gateway]
        Lambda[⚡ Lambda Functions]
    end

    Client[👤 Client] --> API
    API --> Lambda
    Lambda --> CloudApp
    CloudApp --> RDS

    OnPremApp --> VPN
    VPN -.VPN Tunnel.-> VPNGateway
    VPNGateway --> CloudApp
    CloudApp -.Reads.-> OnPremDB
```

## Integration with Code

See language-specific examples for generating deployment diagrams from:

- **Pulumi**: Python, TypeScript, Go IaC code
- **Terraform**: HCL configuration files
- **Docker Compose**: YAML service definitions
- **Kubernetes**: YAML manifests and Helm charts
- **CDK**: CloudFormation templates

Examples: `examples/spring-boot/`, `examples/fastapi/`, etc.

---

**Next Steps:**
- See `activity-diagrams.md` for workflow visualization
- See `code-to-diagram/` for IaC-to-diagram examples
- See language-specific examples for framework-specific patterns
